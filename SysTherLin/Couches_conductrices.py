# Version 1.33 - 2025, May, 14
# Project: SysTherLin (Systèmes thermiques linéaires)
# Copyright (Eric Ducasse 2018)
# Licensed under the EUPL-1.2 or later
# Institution: ENSAM / I2M
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Modélisation de multicouches linéaires 1D plan, ou à symétrie
# cylindrique ou sphérique. Les solutions exactes sont calculées dans
# le domaine de Laplace, avant de revenir dans le domaine temporel par
# transformée de Laplace inverse numérique.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from numpy.linalg import solve
from numpy.fft import rfft,irfft
from scipy.special import erf, ive, kve
from typing import TypeVar, Iterable, Callable, Collection, Optional
from numpy.typing import NDArray
TABFLT = NDArray[np.float64]
ITRFLT = TABFLT | list[float] | tuple[float,...]
FLT = float | ITRFLT
TABCPX = NDArray[np.complex128]
ITRCPX = TABCPX | list[complex] | tuple[complex,...]
CPX = complex | ITRCPX
CLTYPE = ( tuple[str] | tuple[str,Optional[TABCPX]] |
           tuple[str,Optional[float],Optional[TABCPX]] )
#==================== CLASSE COUCHE CONDUCTRICE ========================
class CoucheConductrice:
    """Couche conductrice thermique uniforme 1D."""
    def __init__(self, k: float, rho: float, Cp: float, e: float,
                 Tinit: float=0.0) -> None:
        """ k : conductivité thermique [W/m/K]
            rho : masse volumique [kg/m³]
            Cp : capacité calorifique massique [J/K/kg]
            e : épaisseur de la couche [m].
            Tinit : écart initial de température (supposée uniforme) 
                    par rapport à la température de référence."""
        self.__k: float = k
        self.__r: float = rho
        self.__c: float = Cp
        self.__1sa : float
        self.__updateUnSurAlpha()
        self.__e: float = e
        self.__Ti: float = Tinit
    #-------------------------------------------------------------------
    def copy(self) -> "CoucheConductrice":
        return CoucheConductrice(self.k, self.rho, self.Cp,
                                 self.e, self.Tinit)
    #-------------------------------------------------------------------
    def __updateUnSurAlpha(self):
        """Stockage de 1/alpha = rho*Cp/k [m².s]."""
        self.__1sa = self.__r*self.__c/self.__k
    # attributs en lecture seule :
    @property
    def k(self) -> float: return self.__k
    @property
    def rho(self) -> float: return self.__r
    @property
    def Cp(self) -> float: return self.__c
    @property
    def un_sur_alpha(self) -> float: return self.__1sa
    @property
    def e(self) -> float: return self.__e
    @property
    def tau(self) -> float:
        """Constante de temps [s] : e²/(2*alpha)"""
        return 0.5*self.un_sur_alpha*self.e**2
    @property
    def delta_T_initial(self) -> float:
        """Écart initial de température par rapport à la
           température de référence"""
        return self.__Ti
    @property
    def Tinit(self) -> float: return self.__Ti
    #-------------------------------------------------------------------
    # pour l'affichage par print
    def __str__(self) -> str:
        msg: str = "Couche conductrice de paramètres :"
        for nom,val,u in [
                ["Conductivité", self.k, "W/K/m"],
                ["Masse volumique", self.rho, "kg/m³"],
                ["Capacité calorifique", self.Cp, "J/K/kg"],
                ["Épaisseur", 1000*self.e, "mm"],
                ["Constante de temps", self.tau, "s"]]:
            msg += f"\n\t{nom} : {val:.2f} {u}"
        msg += f"\n\tTempérature initiale : {self.Tinit:.2f} °C"
        return msg 
    #-------------------------------------------------------------------
    # Matrice qui permet de calculer T et phi
    def P(self, s: CPX, x: Optional[float]=None) -> TABCPX:
        """s est un vecteur de valeurs complexes. Pour un s donné,
           matrice P telle que
                 (T(x),phi(x)) = P(x).(a,b) + (Tinit/s,0).
           a correspond à la solution en exp(-sqrt(s/alpha)x).
           b correspond à la solution en exp(-sqrt(s/alpha)(e-x))."""
        if x is None: x = self.e # côté droit
        # au cas où s soit une valeur ou une liste
        ts: TABCPX = np.array(s, dtype=np.complex128) 
        M: TABCPX = np.empty( ts.shape+(2,2), dtype=np.complex128)
        R: TABCPX  = np.sqrt(ts*self.un_sur_alpha)
        X: TABCPX = R*x
        EmX: TABCPX = R*(self.e-x)
        Eplus: TABCPX = np.exp(-X)
        Emoins: TABCPX = np.exp(-EmX)
        M[:,0,0],M[:,0,1] = Eplus,Emoins
        M[:,1,0],M[:,1,1] = R*Eplus,-R*Emoins
        M[:,1,:] *= self.k
        return M
CCP = CoucheConductrice
#============== CLASSE COUCHE CONDUCTRICE CYLINDRIQUE ==================
class CoucheConductriceCylindrique(CoucheConductrice):
    """Couche conductrice thermique uniforme 1D, à symétrie cylindrique.
    """
    def __init__(self, k: float, rho: float, Cp: float, Rmin: float,
                 Rmax: float, Tinit: float=0.0) -> None:
        """ k : conductivité thermique [W/m/K]
            rho : masse volumique [kg/m³]
            Cp : capacité calorifique massique [J/K/kg]
            Rmin : rayon intérieur de la couche [m]
            Rmax : rayon extérieur de la couche [m].
            Tinit : écart initial de température (supposée uniforme) 
                    par rapport à la température de référence."""
        CoucheConductrice.__init__(self, k, rho, Cp, Rmax-Rmin, Tinit)
        self.__r_min: float = Rmin
        self.__r_max: float = Rmax
        self.__r_mid: float = 0.5*(Rmin+Rmax) 
    #-------------------------------------------------------------------
    def copy(self) -> 'CoucheConductriceCylindrique':
        return CoucheConductriceCylindrique(self.k, self.rho, self.Cp,
                                            self.Rmin, self.Rmax,
                                            self.Tinit)
    #-------------------------------------------------------------------
    @property
    def Rmin(self) -> float: return self.__r_min
    @property
    def Rmax(self) -> float: return self.__r_max 
    @property
    def tau(self) -> float:
        """Constante de temps [s] : (formule semi-empirique)"""
        tau0: float = 0.25*self.un_sur_alpha*self.e**2
        rho: float = self.Rmin/self.Rmax
        g: float = 0.30440
        c: float = 0.68012
        l0: float = np.log(g)
        l1: float = np.log(g+1)
        lr: float = np.log(g+rho)
        f: float = c*( 1-rho + (l1-lr)/(l0-l1) )
        return tau0*(1+rho+f)
    #-------------------------------------------------------------------
    # pour l'affichage par print
    def __str__(self) -> str:
        msg: str = "Couche conductrice cylindrique de paramètres :"
        for nom,val,u in [
                ["Conductivité", self.k, "W/K/m"],
                ["Masse volumique", self.rho, "kg/m³"],
                ["Capacité calorifique", self.Cp, "J/K/kg"],
                ["Épaisseur", 1000*self.e, "mm"],
                ["Rayon intérieur", 1000*self.Rmin, "mm"],
                ["Rayon extérieur", 1000*self.Rmax, "mm"],
                ["Constante de temps", self.tau, "s"]]:
            msg += f"\n\t{nom} : {val:.2f} {u}"
        msg += f"\n\tTempérature initiale : {self.Tinit:.2f} °C"
        return msg 
    #-------------------------------------------------------------------                          
    # Matrice qui permet de calculer T et phi
    def P(self, s: CPX, r: Optional[float]=None) -> TABCPX:
        """s est un vecteur de valeurs complexes. Pour un s donné,
           matrice P telle que
                 (T(x),phi(x)) = P(x).(a,b) + (Tinit/s,0).
           a correspond à la solution en K_n (Bessel mod. 2nde espèce).
           b correspond à la solution en I_n (Bessel mod. 1ère espèce).
           """
        #kve(v, z) = kv(v, z) * exp(z)
        #ive(v, z) = iv(v, z) * exp(-abs(z.real))
        if r is None: r = self.Rmax # côté droit
        # au cas où s soit une valeur ou une liste
        ts: TABCPX = np.array(s) 
        M: TABCPX = np.empty( ts.shape+(2,2), dtype=np.complex128)
        B: TABCPX = np.sqrt(ts*self.un_sur_alpha)
        if r <= 0: # Divergence sauf si a = 0
            Z: TABCPX = np.zeros_like(s)
            M[:,0,0],M[:,0,1] = Z,np.exp(-(B*self.__r_mid).real)
            M[:,1,0],M[:,1,1] = Z,Z # Flux nul en r=0 (symétrie)
            return M
        Br: TABCPX = B*r
        Bdr: TABCPX = B*(r-self.__r_mid)
        Eplus: TABCPX = np.exp(-Bdr)
        Emoins: TABCPX = np.exp(Bdr.real)
        M[:,0,0],M[:,0,1] = Eplus*kve(0,Br),Emoins*ive(0,Br)
        M[:,1,0],M[:,1,1] = Eplus*kve(1,Br),-Emoins*ive(1,Br)
        M[:,1,:] = np.einsum("i,ij->ij", self.k*B, M[:,1,:])
        return M
CCC = CoucheConductriceCylindrique
#============== CLASSE COUCHE CONDUCTRICE SPHÉRIQUE ====================
class CoucheConductriceSpherique(CoucheConductrice):
    """Couche conductrice thermique uniforme 1D, à symétrie sphérique.
    """
    def __init__(self, k: float, rho: float, Cp: float, Rmin: float,
                 Rmax: float, Tinit: float=0.0) -> None:
        """ k : conductivité thermique [W/m/K]
            rho : masse volumique [kg/m³]
            Cp : capacité calorifique massique [J/K/kg]
            Rmin : rayon intérieur de la couche [m]
            Rmax : rayon extérieur de la couche [m].
            Tinit : écart initial de température (supposée uniforme) 
                    par rapport à la température de référence."""
        CoucheConductrice.__init__(self, k, rho, Cp, Rmax-Rmin, Tinit)
        self.__r_min: float = Rmin
        self.__r_max: float = Rmax
        self.__r_mid: float = 0.5*(Rmin+Rmax) 
    #-------------------------------------------------------------------
    def copy(self) -> "CoucheConductriceSpherique":
        return CoucheConductriceSpherique(self.k, self.rho, self.Cp,
                                          self.Rmin, self.Rmax,
                                          self.Tinit)
    #-------------------------------------------------------------------
    @property
    def Rmin(self) -> float: return self.__r_min
    @property
    def Rmax(self) -> float: return self.__r_max
    @property
    def tau(self) -> float:
        """Constante de temps [s] : alpha/(6*e²) * (1+2*Rmin/Rmax)"""
        tau0: float = self.un_sur_alpha*self.e**2/6
        rho: float = self.Rmin/self.Rmax
        return tau0*( 1 + 2*rho ) 
    #-------------------------------------------------------------------
    # pour l'affichage par print
    def __str__(self) -> str:
        msg: str = "Couche conductrice sphérique de paramètres :"
        for nom,val,u in [
                ["Conductivité", self.k, "W/K/m"],
                ["Masse volumique", self.rho, "kg/m³"],
                ["Capacité calorifique", self.Cp, "J/K/kg"],
                ["Épaisseur", 1000*self.e, "mm"],
                ["Rayon intérieur", 1000*self.Rmin, "mm"],
                ["Rayon extérieur", 1000*self.Rmax, "mm"],
                ["Constante de temps", self.tau, "s"]]:
            msg += f"\n\t{nom} : {val:.2f} {u}"
        msg += f"\n\tTempérature initiale : {self.Tinit:.2f} °C"
        return msg 
    #-------------------------------------------------------------------                          
    # Matrice qui permet de calculer T et phi
    def P(self, s: CPX, r: Optional[float]=None) -> TABCPX:
        """s est un vecteur de valeurs complexes. Pour un s donné,
           matrice P telle que
                 (T(x),phi(x)) = P(x).(a,b) + (Tinit/s,0).
           a correspond à la solution en exp(-sqrt(s/alpha)(r-Rmin)).
           b correspond à la solution en exp(-sqrt(s/alpha)(Rmax-r)).
           """
        if r is None: r = self.Rmax # côté droit
        # au cas où s soit une valeur ou une liste
        ts: TABCPX = np.array(s) 
        M: TABCPX = np.empty( ts.shape+(2,2), dtype=np.complex128)
        B: TABCPX = np.sqrt(ts*self.un_sur_alpha)
        if r <= 0: # Divergence sauf si a = -b*exp(-B*Rmax)
            L0: TABCPX = B*np.exp(-B*self.Rmax)
            Z: TABCPX = np.zeros_like(ts)
            M[:,0,0],M[:,0,1] = -B,L0
            M[:,1,0],M[:,1,1] = Z,Z # Flux nul en r=0 (symétrie)
            return M
        Br: TABCPX = B*r
        Eplus: TABCPX = np.exp(B*(self.Rmin-r))
        Emoins: TABCPX = np.exp(B*(r-self.Rmax))
        M[:,0,0],M[:,0,1] = Eplus,Emoins
        M[:,0,:] /= r
        M[:,1,0],M[:,1,1] = Eplus*(1+Br),Emoins*(1-Br)
        M[:,1,:] *= self.k/r**2
        return M
CCS = CoucheConductriceSpherique
##################### TESTS ÉLÉMENTAIRES ###############################
if __name__ == "__main__":
    brique: CoucheConductrice = CoucheConductrice(
                                    0.84, 1800.0, 840.0, 0.220, 21.0)
    print(brique)
    couche_cyl: CCC = CoucheConductriceCylindrique(0.84, 1800.0, 840.0,
                                                   0.02, 0.025, 22.0)
    print(couche_cyl)
    print("couche_cyl.P([1.0+2.0j],0.021) :")
    print(couche_cyl.P([1.0+2.0j],0.021))
    couche_sph: CCS = CoucheConductriceSpherique(0.84, 1800.0, 840.0,
                                                 0.02, 0.025, 23.0)
    print(couche_sph)
    print("couche_sph.P([1.0+2.0j],0.021) :")
    print(couche_sph.P([1.0+2.0j],0.021))
#======================= CLASSE MULTICOUCHE ============================
COUCHES = CCP | list[CCP] | tuple[CCP,...]
#-----------------
class Multicouche:
    """Multicouche ne contenant que des couches conductrices."""
    __r: int = 8 # Nombre de pas de temps strictement négatif,
                 #     introduit en raison du filtrage passe-bas
                 #     anti-repliement
    __ar: float = -float(__r)
    __QuatreSurPi: float = 4.0/np.pi # constante
    __MoinsRacineQuatreSurPi: float = -np.sqrt(__QuatreSurPi)
    __br: float = 0.25*np.sqrt(np.pi)*float(__r)
    #-------------------------------------------------------------------
    def __init__(self, couches: COUCHES,
                 x_min: Optional[float]=None) -> None:         
        # Tuple des couches conductrices
        self.__couches: tuple[CCP,...]
        if isinstance(couches, CoucheConductrice):
            self.__couches = (couches,)
        else :
            self.__couches = tuple(couches)
        # Nombre de couches conductrices
        self.__n: int = len(self.__couches) 
        self.__type: str
        prem_couch: CCP = self.__couches[0]
        self.__cyl_or_sph: bool
        if isinstance(prem_couch, CCC):
            self.__cyl_or_sph = True
            self.__type = "cyl"
        elif isinstance(prem_couch, CCS):
            self.__cyl_or_sph = True
            self.__type = "sph"
        else: # couches planes
            self.__cyl_or_sph = False
            self.__type = "pla"
        xmin: float
        msg: str
        if x_min is None:
            if self.__cyl_or_sph:
                xmin = prem_couch.Rmin
            else:
                xmin = 0.0
        else :
            xmin = float(x_min)
            if self.__cyl_or_sph:
                if abs(xmin-prem_couch.Rmin) > 1e-12*prem_couch.e:
                    msg = ("Constructeur de Multicouche :: "
                           + f"attention : x_min [{xmin:.3e}] "
                           + "n'est pas voisin de Rmin ["
                           + f"{prem_couch.Rmin:.3e}] !")
                    print(msg)
        self.__centre: bool
        if self.__cyl_or_sph and abs(xmin)<1e-14*prem_couch.e:
            # Côté gauche confondu avec l'axe/le centre de symétrie
            self.__centre = True
        else:
            self.__centre = False
        X: list[float] = [xmin]
        x: float = xmin
        for nc,couche in enumerate(self.__couches, 1):
            x += couche.e
            if (self.__cyl_or_sph
                and abs(x-couche.Rmax) > 1e-12*couche.e):
                msg = ("Constructeur de Multicouche :: "
                       + f"attention : x [{x:.3e}] n'est pas "
                       + f"voisin de Rmax [{couche.Rmax:.3e}] "
                       + f"dans la couche {nc} !")
                print(msg)                
            X.append(x)
        self.__X: TABFLT  = np.array(X)    # positions des interfaces
        self.__d: float                    # durée de la simulation
        self.__Ts: float                   # période d'échantillonnage
        self.__nt: int                     # nombre de valeurs en temps
        self.__s: Optional[TABCPX] = None  # Vecteur de complexes
                                           # (domaine de Laplace)
        #+++++++++ Matrice du système à résoudre 2n x 2n ++++++++++++
        self.__AB: Optional[TABCPX] = None # coefficients (ai,bi)
        #+++++++++ Côté gauche ++++++++++++
        # Condition limite à gauche
        self.__CLG: Optional[str]
        if self.__centre:
            self.__CLG = "(Symétrie)"
        else:
            self.__CLG = None
        # Fonction représentant la T.L. du signal imposé à gauche
        self.__TLFG: Optional[Callable[[CPX], CPX]] = None
        # Valeurs de la transformée de Laplace de la fonction imposée
        # à gauche, aux instants  positifs (signal entré par
        # l'utilisateur)
        self.__VG: Optional[TABCPX] = None
        # Signal numérique imposé à gauche (obtenu par TLrec)    
        self.__sigG: Optional[TABFLT] = None
        # Coefficient de convection à gauche
        self.__hG: Optional[float] = None
        # Température initiale à gauche
        self.__TiG: float = prem_couch.Tinit   
        #+++++++++ Côté droit ++++++++++++
        # Condition limite à droite
        self.__CLD: Optional[str] = None
        # Fonction représentant la T.L. du signal imposé à droite
        self.__TLFD: Optional[Callable[[CPX], CPX]] = None
        # Valeurs de la transformée de Laplace de la fonction imposée
        # à droite, aux instants positifs (signal entré par
        # l'utilisateur)
        self.__VD: Optional[TABCPX] = None
        # Signal numérique imposé à droite (obtenu par TLrec)
        self.__sigD: Optional[TABFLT] = None
        # Coefficient de convection à droite 
        self.__hD: Optional[float] = None
        # Température initiale à droite
        self.__TiD: float = self.__couches[-1].Tinit
        self.numero: Optional[int]=None
        self.no_cav: Optional[List[int]]=None
    #-------------------------------------------------------------------
    @property
    def nb(self) -> int:
        """Nombre de couches conductrices."""
        return self.__n
    @property
    def X(self) -> TABFLT:
        """Positions des bords et interfaces"""
        return np.array(self.__X.tolist())
    @property
    def Tinit_gauche(self) -> float:
        return self.__TiG
    @property
    def Tinit_droite(self) -> float:
        return self.__TiD
    #-------------------------------------------------------------------
    def definir_temps(self, duree: float, Ts: float) -> None:
        """'duree' est la durée de la simulation et 'Ts' la
           période d'échantillonnage."""
        self.__Ts = Ts
        # Nombre nécessairement pair de pas de temps positifs
        self.__nt = 2*int(np.ceil(0.5*duree/Ts))
        # On rajoute les pas de temps négatifs
        self.__nt += Multicouche.__r
        # Durée totale
        self.__d = self.__nt*Ts
        # Partie réelle commune aux valeurs de la variable de Laplace s
        gamma: float = 11.5 / duree
        # pas de discrétisation en fréquences
        df: float = 1.0 / self.__d
        # Vecteurs des valeurs de la variable de Laplace s
        self.__s = gamma + (2j * np.pi * df
                            * np.arange(0,self.__nt//2+1,1))
        self.__update_VH_VG()
        self.__update_AB() # Calcul, si c'est possible de la solution 
    #-------------------------------------------------------------------
    def definir_signal(self, cote:str,
                       H: Callable[[complex],complex] | ITRFLT) -> None:
        """'cote' vaut 'G' (gauche) ou 'D' (droite).
           'H' est la transformée de Laplace (fonction) du signal,
           ou un signal numérique que l'on filtrera avant de calculer
           sa transformée de Laplace Numérique."""
        if cote.lower() in ["g", "gauche"]:
            if self.__centre:
                print("Multicouche.definir_signal :: attention :"
                      + " Inutile de définir un signal sur le"
                      + " centre de symétrie !")
                self.__TLFG = None
            else:
                if callable(H): # H est une fonction, transformée de
                                # Laplace du signal
                    self.__TLFG = H
                else: # Signal numérique partant de t=0
                    self.__sigG = self.__set_signal(H)
        elif cote.lower() in ["d", "droite"]:
            if callable(H): # H est une fonction, transformée de
                            # Laplace du signal
                self.__TLFD = H
            else: # Signal numérique partant de t=0
                self.__sigD = self.__set_signal(H)
        else:
            print(f'cote "{cote}" inconnu')
            return
        self.__update_VH_VG() 
    #-------------------------------------------------------------------        
    def __set_signal(self, signal: ITRFLT) -> TABFLT:
        deb_msg: str = "Multicouche.definir_signal :: erreur : "
        msg: str
        try:
            sig = np.array(signal)
        except:
            msg = (deb_msg
                   + "Échec de la conversion du signal de type "
                   + f"'{type(signal).__name__}' en ndarray")
            raise ValueError(msg)
        nbval: int = self.__nt - Multicouche.__r
        if sig.shape != (nbval,):
            msg = (deb_msg
                   + "Le signal fourni doit être de forme "
                   + f"{(nbval,)} et non pas {sig.shape} !")
            raise ValueError(msg)
        return np.append(np.zeros(self.__r), sig)
    #-------------------------------------------------------------------                                                                          
    def __update_VH_VG(self) -> None:
        if self.__s is None:
            self.__VG = None
            self.__VD = None
            self.__sigG = None
            self.__sigD = None
            return
        # Prise en compte des températures initiales ici
        if self.__TLFG is None:
            if self.__sigG is None:
                self.__VG = None
            else:
                signalG: TABFLT = self.__sigG[Multicouche.__r:]
                if self.__CLG in ("Dirichlet", "Convection"):
                    signalG -= self.__TiG
                self.__VG = self.TLdir(signalG)
        else:
            if self.__CLG in ("Dirichlet", "Convection"):
                self.__VG = self.__TLFG(self.__s)-self.__TiG/self.__s
            else:
                self.__VG = self.__TLFG(self.__s)
            self.__sigG = self.TLrec(self.__VG)
        if self.__TLFD is None: 
            if self.__sigD is None:
                self.__VD = None
            else:
                signalD: TABFLT = self.__sigD[Multicouche.__r:]
                if self.__CLD in ("Dirichlet", "Convection"):
                    signalD -= self.__TiD
                self.__VD = self.TLdir(signalD)
        else:
            if self.__CLD in ("Dirichlet", "Convection"):
                self.__VD = self.__TLFD(self.__s)-self.__TiD/self.__s
            else:
                self.__VD = self.__TLFD(self.__s)
            self.__sigD = self.TLrec(self.__VD)
        self.__update_AB() # Calcul, si c'est possible de la solution 
    #-------------------------------------------------------------------
    def definir_CL(self, cote: str, CL_type: str,
                   coef_convection: Optional[float]=None) -> None:
        if CL_type.lower() in ["d", "dirichlet"]:
            CL_type = "Dirichlet"
        elif CL_type.lower() in ["n", "neumann"]:
            CL_type = "Neumann"
        elif CL_type.lower() in ["c", "conv", "convec", "convection"]:
            CL_type = "Convection"
            if coef_convection is None:
                print("Coefficient de convection manquant !")
                return
        elif CL_type.lower() in ["s", "(s)", "sym", "(sym)",
                                 "symetrie", "(symetrie)",
                                 "symétrie", "(symétrie)"]:
            CL_type = "(Symétrie)"
        else:
            print(f"Type '{CL_type}' de condition limite non reconnu.")
            return
        if cote.lower() in ["g", "gauche"]:
            if self.__centre and CL_type in ["Dirichlet","Neumann",
                                             "Convection"]:
                print("Multicouche.definir_CL :: attention :"
                      + " Inutile de définir la condition limite "
                      + "au centre : symétrie !")
                CL_type = "(Symétrie)"
            self.__CLG = CL_type
            if CL_type == "Convection":
                self.__hG = coef_convection
            else:
                self.__hG = None
        elif cote.lower() in ["d", "droite"]:
            self.__CLD = CL_type
            if CL_type == "Convection":
                self.__hD = coef_convection
            else:
                self.__hD = None
        else:
            print(f'cote "{cote}" inconnu')
            return
        self.__update_AB() # Calcul, si c'est possible de la solution 
    #-------------------------------------------------------------------
    def __check(self, verbose: bool=False,
                signals: tuple[bool,bool]=(True,True)) -> bool:        
        """Renvoie un booléen qui indique si le calcul est possible."""
        if verbose:
            prt = print # verbose : affichage erreurs
        else:
            def prt(*a,**k): pass
        chk: bool = True
        if self.__s is None:
            prt("Valeurs de la T.L. non définies")
            chk = False
        if signals[0] and not self.__centre and self.__VG is None:
            prt("Signal à gauche non défini.")
            chk = False
        if self.__CLG is None:
            prt("Condition limite à gauche non définie.")
            chk = False
        if signals[1] and self.__VD is None:
            prt("Signal à droite non défini.")
            chk = False
        if self.__CLD is None:
            prt("Condition limite à droite non définie.")
            chk = False
        return chk 
    #-------------------------------------------------------------------
    def tout_est_defini(self,
                        signals: tuple[bool,bool]=(True,True)) -> bool:
        """Vérifie que tout est bien défini."""
        return self.__check(verbose=True, signals=signals) 
    #-------------------------------------------------------------------
    def M_matrix(self, raised_errors: bool=True,
                 verbose: bool=False) -> Optional[TABCPX]:
        msg: str
        if not self.__check(signals=(False,False)): # Calcul impossible
            msg = ("Multicouche.M_matrix :: données insuffisantes"
                   + " pour calculer la matrice M.")
            if raised_errors: raise ValueError(msg)
            elif verbose: print(msg)
            return None
        dn: int = 2*self.nb
        s: TABCPX = self.__s
        couches: tuple[CCP,...] = self.__couches
        M: TABCPX = np.zeros( s.shape+(dn,dn), dtype=np.complex128)
        # Condition limite à gauche
        cm1: CoucheConductrice = couches[0]
        P0: TABCPX = cm1.P(s, self.__X[0])
        P1: TABCPX
        if self.__CLG == "(Symétrie)":
            Z: TABCPX = np.zeros_like(s)
            U: TABCPX = np.ones_like(s)
            if self.__type == "cyl":
                M[:,0,:2] = np.array([U,Z]).transpose() # a=0
            elif self.__type == "sph": # a + exp(-B*e)*b = 0
                M[:,0,:2] = np.array(
                    [U, np.exp(-cm1.e*np.sqrt(s*cm1.un_sur_alpha))]
                    ).transpose()
            else: # impossible
                print("Problème ! CL (Symétrie) dans le cas plan")
                return None
        elif self.__CLG == "Dirichlet":
           M[:,0,:2] = P0[:,0,:]
        elif self.__CLG == "Neumann":
           M[:,0,:2] = P0[:,1,:]
        elif self.__CLG == "Convection":
            M[:,0,:2] = P0[:,0,:] + P0[:,1,:]/self.__hG
        else:
            msg = "CLG '{self.__CLG}' impossible !"
            if raised_errors: raise ValueError(msg)
            else:
                print(msg)
                return None           
        # Continuité aux interfaces
        for i,(c,x) in enumerate( zip(couches[1:],self.__X[1:]) ):
            di: int = 2*i
            if self.__cyl_or_sph:
                P0 = cm1.P(s,x)
                P1 = c.P(s,x)
            else:
                P0 = cm1.P(s)
                P1 = c.P(s,0)
            M[:, di+1:di+3, di:di+2] = P0
            M[:, di+1:di+3, di+2:di+4] = -P1
            cm1 = c
        # Condition limite à droite
        if self.__cyl_or_sph:
            P1 = cm1.P(s,self.__X[-1])
        else:
            P1 = cm1.P(s)
        if self.__CLD == "Dirichlet":
            M[:,-1,-2:] = P1[:,0,:]
        elif self.__CLD == "Neumann":
            M[:,-1,-2:] = P1[:,1,:]
        elif self.__CLD == "Convection":
            M[:,-1,-2:] = P1[:,0,:]-P1[:,1,:]/self.__hD
        else:
            msg = f"CLD '{self.__CLD}' impossible !"
            if raised_errors: raise ValueError(msg)
            else:
                print(msg)
                return None           
        return M 
    #-------------------------------------------------------------------        
    def __update_AB(self) -> None:
        """Déterminer les T.L. des températures et des densités
           surfaciques de flux sur les bords."""
        M: TABCPX = self.M_matrix(raised_errors=False)
        if M is None: return
        # print((abs(M[-1,:,:])>1e-8)*1) # Contrôle de la forme de M
        # Second membre
        S: TABCPX = np.zeros( M.shape[:-1], dtype=np.complex128)
        if not self.__check(): # Calcul impossible
            self.__AB = None
            return
        # Côté gauche
        if not self.__centre:
            S[:,0] = self.__VG
        # Interfaces avec prise en compte des températures initiales
        TinitL: float = self.couches[0].Tinit
        for i,layer in enumerate(self.couches[1:],1):
            TinitR: float = layer.Tinit
            S[:,2*i-1] += (TinitR-TinitL)/self.__s
            TinitL = TinitR
        # Côté droit
        S[:,-1] = self.__VD
        # Résolution
        try : # numpy 1.x
            self.__AB = solve(M,S)
        except : # numpy 2.x
            S.shape = S.shape + (1,)
            self.__AB = solve(M,S)
            self.__AB.shape = self.__AB.shape[:-1]
        return 
    #-------------------------------------------------------------------
    def set_AB(self, new_AB: TABCPX ) -> None:
        shp: tuple[int,...] = self.__s_shape + (2*self.nb,)
        if new_AB.shape == shp:
            self.__AB = new_AB
        else:
            msg: str = ("Multicouche.set_AB :: Erreur : tailles "
                        + f"incompatibles : {new_AB.shape} donné "
                        + f"pour {shp} demandé.")
            raise ValueError(msg) 
        return
    #-------------------------------------------------------------------        
    def T_phi(self, x:float) -> tuple[TABFLT, TABFLT, TABFLT]:
        """ Renvoie les vecteurs 'instants','température' et
            'densité surfacique de flux' à la position x (scalaire)."""
        if self.__AB is None:
            print("Impossible de calculer les champs : "
                  + "coefficients inconnus")
        if x <= self.__X[0]: # x inférieur à la plus petite des valeurs
            x = self.__X[0]
        elif x >= self.__X[-1]: # x supérieur à la plus grande
            x = self.__X[-1]
        xg: float = self.__X[0]
        V: TABCPX
        T: TABCPX
        phi: TABCPX
        P: TABCPX
        for i,(xd,c) in enumerate(zip(self.__X[1:],self.__couches)):
            if x <= xd: # On a trouvé la couche contenant x
                if self.__cyl_or_sph: # Cylindrique ou sphérique
                    P = c.P(self.__s, x)
                else: # Plan
                    P = c.P(self.__s, x-xg)
                V = np.einsum("ijk,ik->ij", P,
                                      self.__AB[:, 2*i: 2*i+2])
                T = V[:,0]
                phi = V[:,1]
                # T et phi désignent la TL des signaux cherchés
                break
            xg = xd
        return (self.timeValues,
                self.TLrec(T) + c.Tinit,
                self.TLrec(phi)) 
    #-------------------------------------------------------------------
    @property
    def timeValues(self) -> TABFLT:
        """Instants auxquels sont calculées les réponses du système."""
        Ts: float = self.__Ts
        t0: float = -float(Multicouche.__r)*Ts   # Instant de début
                                                 # du signal
        return np.arange(self.__nt)*Ts + t0 
    #-------------------------------------------------------------------
    @property
    def positiveTimeValues(self) -> TABFLT:
        return np.arange(self.__nt-Multicouche.__r)*self.__Ts
    #-------------------------------------------------------------------
    @property
    def numberOfNegativeTimeValues(self) -> int:
        return Multicouche.__r
    #-------------------------------------------------------------------
    def signal(self, cote: str) -> TABFLT:
        if cote.lower() in ['g', 'gauche']:
            if self.__centre:
                print("Multicouche.signal :: Condition de symétrie !")
                return np.zeros(self.__nt)
            if self.__CLG in ("Dirichlet", "Convection"):
                return self.__sigG + self.__TiG
            else:
                return self.__sigG
        elif cote.lower() in ['d', 'droite']:
            if self.__CLD in ("Dirichlet","Convection"):
                return self.__sigD + self.__TiD
            else:
                return self.__sigD
        msg: str = (f"Multicouche.signal :: côté '{cote}' inconnu !")
        raise ValueError(msg)
    #-------------------------------------------------------------------
    def TLdir(self, signal: ITRFLT) -> TABCPX:
        msg: str
        if self.__s is None:
            msg = ("Multicouche.TLdir :: Erreur : impossible de "
                   + "calculer une transformée de Laplace "
                   + "numérique\n\tpuisque le temps n'est pas "
                   + "défini. Il faut d'abord appeler la méthode "
                   + "'definir_temps'.")
            raise ValueError(msg)            
        vsignal: TABFLT = np.array(signal, dtype=np.float64)
        nb_val: int = self.__nt - Multicouche.__r
        if vsignal.shape != (nb_val,):
            msg = ("Multicouche.TLdir :: Erreur : le signal est "
                   + f"de forme {vsignal.shape} et non pas "
                   + f"{(nb_val,)}\n\t(signal causal de 0.0 à "
                   + f"{(nb_val-1)*self.__Ts:.5e} s "
                   + f"par pas de {self.__Ts:.3e} s")
            raise ValueError(msg)        
        gamma: float = self.__s[0].real
        vsignal= np.append(vsignal, np.zeros(Multicouche.__r))
        mt0: float = Multicouche.__r*self.__Ts
        vsignal *= np.exp(-gamma*(self.timeValues+mt0))        
        return rfft(vsignal) * self.__Ts 
    #-------------------------------------------------------------------
    def TLrec(self, U: TABCPX) -> TABFLT :
        """Renvoie les valeurs du signal numérique dont la T.L.
           est le vecteur U (pour les valeurs s dans self.__s."""
        if U.shape != self.__s.shape:
            msg: str = ("Anomalie : U et s n'ont pas la même dimension"
                        + f" :\n\t{U.shape} / {self.__s.shape}")
            raise ValueError(msg)
        # Filtrage passe-bas pour la discontinuité en t=0
        # Constantes : 
        a: float = Multicouche.__QuatreSurPi
        ar: float = Multicouche.__ar
        b: float = Multicouche.__MoinsRacineQuatreSurPi
        br: float = Multicouche.__br
        Sr: TABCPX = self.__Ts * self.__s
        Y: TABCPX =  0.5*U*np.exp(Sr*(a*Sr+ar))*(1+erf(b*Sr+br))
        Y[-1] = 0.5*Y[-1].real
        gamma: float = self.__s[0].real
        mt0: float = float(Multicouche.__r) * self.__Ts
        return (irfft(Y).real / self.__Ts
                * np.exp(gamma*(self.timeValues+mt0)))
    #-------------------------------------------------------------------
    def TLsig(self, cote: str, offset: bool=True) -> TABCPX:
        if cote.lower() in ["g", "gauche"]:
            if self.__centre:
                print("Multicouche.signal :: Condition de symétrie !")
                return np.zeros_like(self.__s)
            if offset and self.__CLG in ("Dirichlet", "Convection"):
                return self.__VG + self.__TiG/self.__s
            else:
                return self.__VG
        elif cote.lower() in ["d", "droite"]:
            if offset and self.__CLD in ("Dirichlet", "Convection"):
                return self.__VD + self.__TiD/self.__s
            else:
                return self.__VD
        msg: str = (f"Multicouche.TLsig :: Erreur : côté '{cote}' "
                    + "inconnu")
        raise ValueError(msg)
    #-------------------------------------------------------------------        
    def __str__(self) -> str:
        tp: str
        if self.__type == "plan": tp = "plane(s)"
        elif self.__type == "cyl": tp = "cylindrique(s)"
        elif self.__type == "sph": tp = "sphérique(s)"
        else: tp = "" # (erreur)
        msg: str = (f"Multicouche à {self.__n} couche(s) "
                    + f"conductrice(s) {tp}\n")
        msg += "\td'épaisseur(s) en millimètres : ["
        for c in self.__couches: msg += f"{1000*c.e:.1f}, "
        msg = msg[:-2]+"]"
        msg += f'\n\tCondition limite à gauche : "{self.__CLG}"'
        msg += f'\n\tCondition limite à droite : "{self.__CLD}"'
        return msg 
    #-------------------------------------------------------------------
    @property
    def couches(self) -> tuple[CoucheConductrice,...]:
        return self.__couches
    @property
    def geometrie(self) -> str:
        if self.__type == "plan": return "plane"
        if self.__type == "cyl": return "cylindrique"
        if self.__type == "sph": return "sphérique"
    @property
    def CLG(self) -> CLTYPE:
        if self.__centre: return ("(Symétrie)",)
        if self.__CLG == "Convection":
            return ("Convection",self.__hG,self.__VG)
        elif self.__CLG in ["Dirichlet","Neumann"]:
            return (self.__CLG,self.__VG)
        else:
            return ("Non définie",)
    @property
    def CLD(self) -> CLTYPE:
        if self.__CLD == "Convection":
            return ("Convection",self.__hD,self.__VD)
        elif self.__CLG in ["Dirichlet","Neumann"]:
            return (self.__CLD,self.__VD)
        else:
            return ("Non définie",)
    @property
    def s(self) -> TABCPX: return self.__s
    @property
    def __s_shape(self) -> tuple[int,...]:
        if self.__s is None: return (0,)
        else: return self.__s.shape
    @property
    def ns(self) -> int: return  self.__s_shape[0]
MC = Multicouche
##################### TESTS ÉLÉMENTAIRES ###############################
if __name__ == "__main__":
    #++++++++++++++++++++++++
    choix: int = 3 # 1,2,3,4
    #++++++++++++++++++++++++
    nb: int
    C: MC
    Rmin: float
    Rmax: float
    VR: TABFLT
    couches: list[CCP]
    if choix == 1: ## Exemple de multicouche plan 
        # Étape 1 - Définition du multicouche
        nb = 4 # nb couches identiques d'épaisseur total 22 cm
        petite_brique: CoucheConductrice = CoucheConductrice(
                            0.84, 1800.0, 840.0,0.220/nb, Tinit=20.0)
        C = Multicouche(nb*[petite_brique])
    elif choix == 2: ## Exemple de multicouche cylindrique
        # Étape 1 - Définition du multicouche
        Rmin = 0.0
        Rmax = Rmin + 0.220
        nb = 4
        VR = np.linspace(Rmin, Rmax, nb+1)
        couches = []
        for rmin,rmax in zip(VR[:-1],VR[1:]):
            couches.append(
                CoucheConductriceCylindrique(0.84, 1800.0, 840.0, rmin,
                                             rmax, Tinit=20.0))
        C = Multicouche(couches)
    elif choix == 3: ## Exemple de multicouche sphérique
        # Étape 1 - Définition du multicouche
        Rmin = 0.5
        Rmax = Rmin+0.220
        nb = 3
        VR = np.linspace(Rmin,Rmax,nb+1)
        couches = []
        for rmin,rmax in zip(VR[:-1],VR[1:]):
            couches.append(
                CoucheConductriceSpherique(0.84, 1800.0, 840.0, rmin,
                                           rmax, Tinit=20.0))
        C = Multicouche(couches)
    if choix == 4: ## Exemple de multicouche plan avec températures
                   ##  initiales différentes : Versions >= 1.2
        brique_chaudeG: CoucheConductrice = CoucheConductrice(
                                0.84, 1800.0, 840.0, 0.05, Tinit=70.0)
        brique_froide: CoucheConductrice = CoucheConductrice(
                                0.84, 1800.0, 840.0, 0.14, Tinit=10.0)
        brique_chaudeD: CoucheConductrice = CoucheConductrice(
                                0.84, 1800.0, 840.0, 0.03, Tinit=50.0)
        C = Multicouche([brique_chaudeG, brique_froide,
                             brique_chaudeD])
    # Étape 2 - Définition de la durée et de la période
    #           d'échantillonnage
    if choix != 4:
        C.definir_temps(2.0e5, 0.5e3)
    else:
        C.definir_temps(2.0e4, 10.0)
    # Étape 3 - Définition des conditions aux limites
    # 3a/ À gauche
    if choix != 2: # pour choix == 2, condition de symétrie
        C.definir_CL("G", "Neumann")
    # Transformée de Laplace du signal imposé : ici signal nul
        C.definir_signal("G", np.zeros_like)
    # 3b/ À droite
    if choix != 4:
        C.definir_CL("D", "Dirichlet")
    # Température imposée : on définit ici les valeurs du signal
        instants_positifs: TABFLT = C.positiveTimeValues
        sig_num = 15.0 + 5*np.cos(instants_positifs*1e-4)
        C.definir_signal("D", sig_num)
        print(C)
    else:
        C.definir_CL("D", "Neumann")
        C.definir_signal("D", np.zeros_like)
    # Étape 4 - Résolution automatique dès que le problème est complet
    # Étape 5 - Tracés
    if C.tout_est_defini():
        import matplotlib.pyplot as plt
        fig = plt.figure(
            "Température et densités surfaciques de flux",
            figsize=(12,7))
        ax_temp,ax_phi = fig.subplots(2,1)
        opt: dict = {"size":14, "family":"Arial", "weight":"bold"}
        plt.subplots_adjust(left=0.075, right=0.99, bottom=0.075,
                            top=0.99, hspace=0.15)
        ax_phi.set_xlabel(r"Instant $t$ $[h]$", **opt)
        colors: list[tuple[float,float,float]] = [
            (0.8,0.,0.), (0.,0.5,0.), (0.,0.,1.), (0.8,0.8,0),
            (1.,0.,1.), (0.,0.7,0.7), (0.,0.3,0.6), (0.6,0.3,0.),
            (0.8,0.3,0.3), (0.4,0,0.8), (0.6,0.6,0.6)]
        nbcol: int = len(colors)
        nocol: int = 0
        phi_lbl: str = r"Dens. surf. de flux $\phi(t)$"
        X: TABFLT
        coef: float
        if choix != 4:
            X = np.linspace(C.X[0], C.X[-1], 5)
            line_styles = 5*["-"]
            coef = 1.0
            phi_lbl += r" [W$\cdot$m²]"
        else:
            X = np.array([C.X[0],0.05*C.X[0]+0.95*C.X[1],
                          C.X[1]-1e-4, C.X[1]+1e-4,
                          0.95*C.X[1]+0.05*C.X[2],
                          0.5*(C.X[1]+C.X[2]),
                          0.05*C.X[1]+0.95*C.X[2],
                          C.X[2]-1e-4, C.X[2]+1e-4,
                          0.95*C.X[2]+0.05*C.X[3],
                          C.X[3]])
            line_styles = 3*["--"] + 5*["-"] + 3*[":"]
            coef = 1.0e-4
            phi_lbl += r" [W$\cdot$cm²]"
        for x,clr,ls in zip(X, colors, line_styles):
            Vt,T,phi = C.T_phi(x)
            Vh: TABFLT = Vt/3600.0
            ax_temp.plot(Vh, T, ls, color=clr, linewidth=1.6,
                         label=f"$r={1e3*x:.1f}\,$mm")
            ax_phi.plot(Vh, coef*phi, ls, color=clr, linewidth=1.6,
                        label=f"$r={1e3*x:.1f}\,$mm")
        ax_temp.set_ylabel(r"Température $T(t)$ [°C]", **opt)
        ax_temp.grid() ; ax_phi.grid()
        ax_temp.legend(loc="center right",fontsize=12)       
        ax_phi.set_ylabel(phi_lbl, **opt)
        plt.show()
