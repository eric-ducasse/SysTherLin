# Version 1.32 - 2025, May, 12
# Project : SysTherLin (Systèmes thermiques linéaires)
# Copyright (Eric Ducasse 2018)
# Licensed under the EUPL-1.2 or later
# Institution : ENSAM / I2M
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from numpy.linalg import solve
from numpy.fft import irfft
from scipy.special import erf
from typing import TypeVar, Iterable, Callable, Collection, Optional
from Couches_conductrices import (Multicouche, MC, NDArray, TABFLT,
                                  ITRFLT, FLT, TABCPX, ITRCPX, CPX,
                                  CLTYPE)
#==================== CLASSE CAVITÉ ====================================
class Cavite :
    """Cavité dont les parois sont des multicouches 1D."""
    #-------------------------------------------------------------------
    def __init__(self, volume: float, rho: float, Cp: float,
                 parois: list[tuple[MC,str,float]], Tinit: float=0.0):
        """ 'parois' est une liste de triplets (MC,côté,surface).
            'MC' est une instance de la classe Multicouche.
            'côté' vaut 'G' ou 'D' pour indiquer le côté de
            raccordement. 'surface' est la surface de contact entre
            la paroi et la cavité.
            'Tinit' désigne l'écart initial de température de la cavité
            par rapport à la température de référence.
            """
        self.__r: float = rho
        self.__c: float = Cp
        self.__v: float = volume
        self.__a: float = 1.0/(volume*rho*Cp)
        self.__Ti: float = Tinit
        self.__parois: list[MC] = [] # Multicouches dont une paroi est
                                     # en contact avec la cavité
        self.__S: list[float] = []   # Surfaces de contact
        self.__n: list[int] = []     # Côtés des parois concernées
        try :
            msg = ""
            for mc,GD,S in parois :
                if GD.lower() in ["g","gauche"] :
                    CLG: CLTYPE = mc.CLG
                    if CLG[0] != "Convection" :
                        msg = ("Constructeur de Cavite :: erreur : le "
                               + "côté gauche du multicouche :\n"
                               + mc.__str__()
                               + "\n n'a pas une C.L. de convection.")
                        raise
                    self.__n.append(0)
                elif GD.lower() in ["d","droite"] :
                    CLD: CLTYPE = mc.CLD
                    if CLD[0] != "Convection" :
                        msg = ("Constructeur de Cavite :: erreur : le "
                               + "côté droit du multicouche :\n"
                               + mc.__str__()
                               + "\n n'a pas une C.L. de convection.")
                        raise
                    self.__n.append(-1)
                else :
                    msg = ("Constructeur de Cavite :: erreur : le côté"
                           + " du multicouche n'est pas bien spécifié"
                           + f" :\n\t'G' ou 'D' et non pas '{GD}'")
                    raise
                self.__parois.append(mc)
                self.__S.append(S)
        except :
            if msg == "" :
                msg = ("Constructeur de Cavite :: erreur : "
                       + "'parois' n'est pas une liste de "
                       + "triplets (MC,côté,surface)")
            raise ValueError(msg)
    #-------------------------------------------------------------------
    @property
    def rho(self) -> float: return self.__r
    @property
    def Cp(self) -> float: return self.__c
    @property
    def volume(self) -> float: return self.__v
    @property
    def a(self) -> float:
        """T'(t) = a*flux total entrant."""
        return self.__a 
    @property
    def Tinit(self) -> float: return self.__Ti
    @property
    def parois(self) -> tuple[MC,...]: return tuple(self.__parois)
    @property
    def surfaces(self) -> tuple[float,...]: return tuple(self.__S)
    @property
    def cotes(self) -> tuple[int,...]:
        """0 pour gauche et -1 pour droite."""
        return tuple(self.__n)
    #-------------------------------------------------------------------
    def __str__(self) -> str:
        nb_par: int = len(self.parois)
        msg: str
        if nb_par == 1 :
            msg = "Cavité à une seule paroi :"
        else :
            msg = f"Cavité à {nb_par} parois :"
        volume: float
        unitVol: str
        if self.__v >= 1.0:
            volume = self.__v
            unitVol = "m³"
        elif self.__v >= 1.0e-3:
            volume = 1e3*self.__v
            unitVol = "l"
        elif self.__v >= 1.0e-6:
            volume = 1e6*self.__v
            unitVol = "cm³"
        else:
            volume = 1e9*self.__v
            unitVol = "mm³"
        
        for nom,val,u in [
                ["Volume", volume,unitVol],
                ["Masse volumique", self.rho, "kg/m³"],
                ["Capacité calorifique", self.Cp, "J/K/kg"] ]:
            msg += f"\n\t{nom} : {val:.2f} {u}"
        msg += ("\n\tTempérature initiale : "
                + f"{self.Tinit:.2f} °C")
        for no,S in enumerate(self.surfaces,1) :
            if S >= 0.1 :
                msg += (f"\n\tSurface de la paroi n°{no} : "
                        + f"{S:.3f} m²")
            else :
                msg += (f"\n\tSurface de la paroi n°{no} : "
                        + f"{1e4*S:.2f} cm²")
        return msg         
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__" :
    from Couches_conductrices import CoucheConductrice
    ### Exemple d'un aquarium parallélépipède rectangle
    # 1 - Socle
    inox = CoucheConductrice(16.5, 8000.0, 500.0, 3.0e-3, 20.0)
    verre_bas =  CoucheConductrice(1.0, 2800.0, 1000.0, 1.0e-3, 20.0)
    socle = Multicouche([inox, verre_bas]) # Couches de gauche à droite
    socle.definir_CL("G", "Neumann") # Chauffage par le socle
    # 100 W/m² constant -> transformée de Laplace s |-> 100/s
    socle.definir_signal("G", lambda s : 100/s)
    # Convection vers l'intérieur de l'aquarium (eau)
    socle.definir_CL("D", "Convection", 200.0)
    print("Socle :",socle)
    # 2 - Autres parois (côtés et dessus)
    verre =  CoucheConductrice(1.0, 2800.0, 1000.0, 8.0e-3, 20.0)
    coque = Multicouche([verre]) # Une seule couche
    # Convection vers l'intérieur de l'aquarium (eau)
    coque.definir_CL("G","Convection",200.0)
    # Convection vers l'extérieur (air ambiant)
    coque.definir_CL("D","Convection",10.0)
    # Température extérieure 20 constante -> tr. Laplace s |-> 20/s
    coque.definir_signal("D",lambda s : 20/s) # 
    print("Coque :",coque)
    # 3 - cavité
    long, larg, haut = 1.0, 0.4, 0.5
    S_socle = long*larg
    S_coque = long*larg + 2*(long+larg)*haut
    cavite = Cavite(0.2, 1000.0, 4200.0,
                    [(socle,"D",S_socle),(coque,"G",S_coque)],
                    20.0)
    print(cavite)
#================ PETITE CLASSE UTILE ==================================
class orbites :
    #-------------------------------------------------------------------
    def __init__(self, size: int):
        self.__nb: int = size
        self.__orb: list[list[int]] = [[i] for i in range(size)]
    #-------------------------------------------------------------------
    def add(self, pair: list[int]) -> None:
        n0,n1 = pair
        if not (0 <= n0 < self.__nb and 0 <= n1 < self.__nb) :
            return
        for i,e in enumerate(self.__orb):
            if n0 in e : i0 = i
            if n1 in e : i1 = i
        if i0 == i1 :
            return # No new connection
        # Orbits i0 and i1 are merged:
        self.__orb[i0].extend(self.__orb[i1])
        self.__orb[i0].sort()
        self.__orb.pop(i1)
    #-------------------------------------------------------------------
    @property
    def orbites(self) -> tuple[tuple[int,...],...]:
        return tuple([tuple(e) for e in self.__orb])        
    @property
    def nb(self) -> int: return len(self.__orb)
    @property
    def all_connected(self) -> bool: return len(self.__orb) == 1
#================ CLASSE SYSTÈME THERMIQUE LINÉAIRE ====================
ELEMTS = MC | Cavite | list[Cavite] | tuple[Cavite,...]
class SystemeThermiqueLineaire :
    """Système composé uniquement de multicouches 1D conductrices et
       de cavités."""
    #-------------------------------------------------------------------
    def __init__(self, duree: float, dt: float, elements: ELEMTS,
                 calculer: bool=False) :
        """elements est soit un seul multicouche, soit une seule cavité,
           soit une liste de plusieurs cavités dont les parois sont des
           multicouches."""
        self.__type: str
        self.__LMC : list[MC]
        self.__LCAV: list[Cavite]
        self.__T_cav: Optional[list[TABFLT]]
        self.__time: TABFLT
        self.__nneg: int
        self.__s : TABCPX
        self.__bornes: Optional[list[int]]
        self.__dim: Optional[int]
        msg: str
        if isinstance(elements, Multicouche) : # 1 seul multicouche
            self.__type = "un seul multicouche"
            MC = elements
            self.__LMC = [MC]
            MC.definir_temps(duree, dt)
            self.__LCAV = []
            self.__T_cav = []
            self.__time = MC.timeValues
            self.__nneg = MC.numberOfNegativeTimeValues
            self.__s = MC.s
            if calculer : self.calculer_maintenant()
            self.__bornes, self.__dim = None, None
        elif isinstance(elements,Cavite) : # 1 cavité avec des parois
            self.__type = "une seule cavité"
            CAV = elements
            self.__LCAV = [CAV]
            self.__LMC = CAV.parois
            self.__bornes, pos = [0],0
            for mc in self.__LMC :
                mc.definir_temps(duree,dt)
                pos += 2*mc.nb
                self.__bornes.append(pos)
            self.__dim = pos + 1 # Taille de la matrice
            self.__time = mc.timeValues # Commun à tous
            self.__nneg = mc.numberOfNegativeTimeValues
            self.__s = mc.s # idem
            if calculer : self.calculer_maintenant()
        else : # Plusieurs cavités
            self.__type = "plusieurs cavités"
            self.__LCAV = list(elements)
            self.__LMC = [] # Liste des multicouches
            no_mc: int = 0 # Numéro de multicouche
            # On attache « à la volée » à chaque multicouche des
            # attributs publics supplémentaires : 
            #      numero = son rang dans la liste globale
            #      no_cav = couple de booléens avec False si connexion
            #               avec une cavité et True sinon
            #               (numéro de cavité de façon temporaire)
            orb = orbites(len(self.__LCAV))
            for p,cav in enumerate(self.__LCAV) :
                for mc,gd in zip(cav.parois,cav.cotes) :
                    if mc not in self.__LMC :
                        # on numérote le multicouche
                        mc.numero = no_mc
                        if gd == 0 : mc.no_cav = [p,-1]
                        else : mc.no_cav = [-1,p]
                        no_mc += 1
                        self.__LMC.append(mc)
                    else :
                        if mc.no_cav[gd] == -1 :
                            mc.no_cav[gd] = p
                            orb.add(mc.no_cav)
                        else :
                            msg = ("Constructeur de SystemeThermiqueLi"
                                   + "neaire :: Erreur : deux cavités "
                                   + "du même côté du multicouche :\n")
                            msg += mc.__str__()
                            raise ValueError(msg)
            if not orb.all_connected :
                 msg = ("Constructeur de SystemeThermiqueLineaire ::"
                        + f" Attention : {orb.nb} sous-systèmes"
                        + " découplés")
            for mc in self.__LMC : # Conversion en booléens
                for i in [0,1] : mc.no_cav[i] = (mc.no_cav[i] == -1)
            self.__bornes,pos = [0],0
            for mc in self.__LMC :
                mc.definir_temps(duree,dt)
                pos += 2*mc.nb
                self.__bornes.append(pos)
            self.__dim = pos + len(self.__LCAV) # Taille de la matrice
            self.__time = mc.timeValues # Commun à tous
            self.__nneg = mc.numberOfNegativeTimeValues
            self.__s = mc.s # idem
            self.__T_cav = None # Liste des signaux de température dans
                                # les cavités, à calculer
            if calculer : self.calculer_maintenant()
    #-------------------------------------------------------------------
    def calculer_maintenant(self, verbose=False, quiet=False) -> None:
        """Calcule la solution, si cela est possible."""
        if quiet : prt_q = lambda *p:None
        else : prt_q = print
        if verbose: prt = print
        else: prt = lambda *p:None
        prt_q("SystemeThermiqueLineaire.calculer_maintenant...")
        if self.__type == "un seul multicouche" :
            MC = self.__LMC[0]
            prt("+++ Multicouche 1 :")
            if MC.tout_est_defini() :                
                prt_q("... Calcul déjà effectué.")
                # tout est déjà calculé dans le multicouche
        elif self.__type == "une seule cavité" :
            s,bornes = self.__s,self.__bornes
            ns,dm = len(s),self.__dim
            big_M = np.zeros( (ns,dm,dm), dtype=complex)
            big_V = np.zeros( (ns,dm), dtype=complex)
            CAV = self.__LCAV[0]
            Tinit = CAV.Tinit
            OK = True
            for i,(mc,d,f,gd,S) in enumerate(zip(CAV.parois,\
                                             bornes[:-1],bornes[1:],\
                                             CAV.cotes,CAV.surfaces),\
                                             1) :
                prt(f"+++ Multicouche {i} :")
                sgn = [True,True];sgn[gd]=False
                if not mc.tout_est_defini(signals=sgn) :
                    OK = False
                    continue
                big_M[:,d:f,d:f] = mc.M_matrix()
                layer = mc.couches[gd]
                if gd == 0 : # côté gauche
                    LP = layer.P(s,mc.X[0])[:,1,:] # DS Flux
                else : # côté droit
                    LP = layer.P(s)[:,1,:] # DS Flux
                P1 = np.einsum("i,ij->ij", CAV.a*S/s,LP)
                if gd == 0 : # côté gauche
                    big_M[:,d,-1] = -1.0 # Température
                    big_V[:,d] = (Tinit-mc.Tinit_gauche)/s
                                 # Saut initial de Température
                    big_M[:,-1,d:d+2] = -P1  # DS Flux
                    # autre côté :
                    big_V[:,f-1] = mc.TLsig("D", offset=False) 
                else : # côté droit
                    big_M[:,f-1,-1] = -1.0 # Température
                    big_V[:,f-1] = (Tinit-mc.Tinit_droite)/s
                                   # Saut initial de Température
                    big_M[:,-1,f-2:f] = P1  # DS Flux
                    # autre côté :
                    if mc.CLG[0] == '(Symétrie)' :
                        big_V[:,d] = 0.0
                    else :
                        big_V[:,d] = mc.TLsig("G",offset=False)
                #!!!
                # Bogue rectifié à partir de la version 1.23 :
                #       sauts internes de températures initiales
                TinitL = mc.couches[0].Tinit
                for nlay,layer in enumerate(mc.couches[1:],1) :
                    TinitR = layer.Tinit
                    big_V[:,d+2*nlay-1] += (TinitR-TinitL)/self.__s
                    TinitL = TinitR
                #!!!
                prt("\tOK")
            big_M[:,-1,-1] = -1.0
            if OK :
                V_AB = solve(big_M,big_V)
                for mc,d,f in zip(CAV.parois,bornes[:-1],bornes[1:]) :
                    mc.set_AB(V_AB[:,d:f])            
                self.__T_cav = [CAV.Tinit + mc.TLrec(V_AB[:,-1])]
                prt_q("... Calcul effectué.")
        elif self.__type == "plusieurs cavités" :
            s,bornes = self.__s,self.__bornes
            ns,dm = len(s),self.__dim
            big_M = np.zeros( (ns,dm,dm), dtype=complex)
            big_V = np.zeros( (ns,dm), dtype=complex)
            OK = True
            ############
            for p,cav in enumerate(self.__LCAV, 1) :
                prt(f"+++ Cavité {p} :")
                Tinit = cav.Tinit
                for mc,gd,S in zip(cav.parois, cav.cotes, cav.surfaces):
                    no = mc.numero
                    np1 = no + 1
                    d,f = bornes[no:no+2]
                    prt(f"+++ Multicouche {np1} :")
                    if not mc.tout_est_defini(signals=mc.no_cav) :
                        OK = False
                        continue
                    big_M[:,d:f,d:f] = mc.M_matrix()
                    layer = mc.couches[gd]
                    if gd == 0 : # côté gauche
                        LP = layer.P(s,mc.X[0])[:,1,:]  # DS Flux
                    else : # côté droit
                        LP = layer.P(s)[:,1,:]  # DS Flux
                    P1 = np.einsum("i,ij->ij", cav.a*S/s, LP)
                    if gd == 0 : # côté gauche
                        big_M[:,d,-p] = -1.0
                        big_V[:,d] = (Tinit-mc.Tinit_gauche)/s
                                     # Saut initial de Température
                        big_M[:,-p,d:d+2] = -P1
                        # autre côté :
                        if mc.no_cav[-1] :
                            big_V[:,f-1] = mc.TLsig("D", offset=False)
                    else : # côté droit
                        big_M[:,f-1,-p] = -1.0
                        big_V[:,f-1] = (Tinit-mc.Tinit_droite)/s
                                       # Saut initial de Température
                        big_M[:,-p,f-2:f] = P1
                        # autre côté :
                        if mc.no_cav[0] :                            
                            if mc.CLG[0] == '(Symétrie)' :
                                big_V[:,d] = 0.0
                            else :
                                big_V[:,d] = mc.TLsig("G", offset=False)
                    prt("\tOK")
                big_M[:,-p,-p] = -1.0
            ############
            #!!!
            # Bogue rectifié à partir de la version 1.23 :
            #       sauts internes de températures initiales
            for i,mc in enumerate(self.__LMC,1) :
                # Différents multicouches
                prt_q(f"Sauts de températures dans le multicouche {i}")
                no = mc.numero
                d = bornes[no]
                TinitL = mc.couches[0].Tinit
                for nlay,layer in enumerate(mc.couches[1:],1) :
                    TinitR = layer.Tinit
                    big_V[:,d+2*nlay-1] += (TinitR-TinitL)/s
                    TinitL = TinitR
            #!!!
            if OK :
                self.big_M = big_M
                self.big_V = big_V
                V_AB = solve(big_M,big_V) # Résolution du système
                for mc in self.__LMC : # Différents multicouches
                    no = mc.numero
                    d,f = bornes[no:no+2]
                    mc.set_AB(V_AB[:,d:f])
                # Températures dans les cavités
                self.__T_cav = [cav.Tinit + mc.TLrec(V_AB[:,-d])
                                for d,cav in enumerate(self.__LCAV, 1)]
                prt_q("... Calcul effectué.")
        else :
            print("SystemeThermiqueLineaire.calculer_maintenant ::"
                  + f"Type de système '{self.__type}' inconnu!")
        return           
    #-------------------------------------------------------------------        
    @property
    def timeValues(self) -> TABFLT:
        return self.__time
    @property
    def positiveTimeValues(self) -> TABFLT:
        return self.__time[self.__nneg:]
    @property
    def multicouches(self) -> list[MC]:
        return self.__LMC
    @property
    def cavites(self) -> list[Cavite]:
        return self.__LCAV
    @property
    def T_cavites(self) -> Optional[list[TABFLT]]:
        return self.__T_cav
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__" :
    import matplotlib.pyplot as plt
    #####################
    choix = 5  # (1 à 5)
    #####################
    #-------------------------------------------------------------------
    if choix == 1 : # Problème mal posé
        essai1 = SystemeThermiqueLineaire(1800.0, 1.0, socle,
                                          calculer=True)
    #-------------------------------------------------------------------
    elif choix == 2 : # Un multi-couche seul
        socle.definir_signal("D",lambda s: 15.0/s)
        essai2 = SystemeThermiqueLineaire(1800.0, 0.5, socle,
                                          calculer=True)
        mc = essai2.multicouches[0]
        Vt,T,phi = mc.T_phi(mc.X[-1])
        fig = plt.figure("Températures et flux", figsize=(10,6))
        ax_temp,ax_flux = fig.subplots(2,1)
        fig.subplots_adjust(0.08, 0.09, 0.98, 0.99)
        ax_temp.plot(essai2.timeValues, T, "-m", label="$T(t)$")
        ax_temp.grid() ; ax_temp.legend()
        ax_temp.set_ylabel("Température $T(t)$ [°C]")
        ax_flux.plot(essai2.timeValues, phi, "-b", label="$\phi(t)$")
        ax_flux.grid() ; ax_flux.legend()
        ax_flux.set_xlabel("Instants $t$ [s]")
        ax_flux.set_ylabel("$\phi(t)$ [W/m²]")
        plt.show()
    #-------------------------------------------------------------------
    elif choix == 3 : # Exemple de l'aquarium (voir ci-dessus :
                      # lignes 119 à 140
        essai3 = SystemeThermiqueLineaire(80*3600, 10.0, cavite,
                                          calculer=True)
        Theures = essai3.timeValues/3600
        fig = plt.figure("Températures dans l'aquarium", figsize=(10,6))
        ax = fig.subplots(1,1)
        ax.plot(Theures, essai3.T_cavites[0], "-m",
                label=r"$T_{\mathrm{eau}}$")
        socl = essai3.cavites[0].parois[0]
        _,T1,_ = socl.T_phi(0.5*socl.X[-1])
        ax.plot(Theures, T1, "-r", label=r"$T_{\mathrm{socle}}$")
        coqu = essai3.cavites[0].parois[1]
        _,T3,_ = coqu.T_phi(0.5*coqu.X[-1])
        ax.plot(Theures, T3, "-b", label=r"$T_{\mathrm{coque}}$")
        ax.legend(loc="best", fontsize=10)
        ax.grid() ; plt.show()
    #-------------------------------------------------------------------
    elif choix == 4 : # Cylindre d'acier
        from Couches_conductrices import CoucheConductriceCylindrique
        acier = CoucheConductriceCylindrique(50.2, 7.85e3, 1000.0, 0.0,
                                             0.05, 50.0)
                                                   # T° initiale 50°C
        cylindre = Multicouche([acier])
        cylindre.definir_CL("D", "Convection", 200.0) # Plongé dans l'eau
        cylindre.definir_signal("D", lambda s: 10/s)  # T° ext. 10°C
        essai4 = SystemeThermiqueLineaire(2*3600, 10.0, cylindre,
                                          calculer=True)
        Vt,T_ext,F_ext = cylindre.T_phi(0.05)
        _,T_mil,F_mil = cylindre.T_phi(0)
        fig = plt.figure("Températures et flux", figsize=(10,6))
        ax_temp,ax_flux = fig.subplots(2,1)
        fig.subplots_adjust(0.08, 0.09, 0.98, 0.99)
        Vmn = Vt/60
        ax_temp.plot(Vmn, T_ext, "-b", label=r"$T_{\mathrm{ext}}$")
        ax_temp.plot(Vmn, T_mil, "-m", label=r"$T_{\mathrm{mil}}$")
        ax_temp.legend(loc="best", fontsize=10)
        ax_temp.set_ylabel("Température $T(t)$ [°C]")
        ax_flux.plot(Vmn, F_ext, "--r", label=r"$\phi_{\mathrm{ext}}$")
        ax_flux.legend(loc="best",fontsize=10)
        ax_flux.set_xlabel("Instants $t$ [s]")
        ax_flux.set_ylabel("$\phi(t)$ [W/m²]")
        ax_temp.grid() ; ax_flux.grid()
        plt.show()
    #-------------------------------------------------------------------
    elif choix == 5 : # Cylindre d'acier à 50°C plongé dans une cavité
                      #  d'eau à 10°C, avec un air extérieur à 20°C
        from Couches_conductrices import CoucheConductriceCylindrique
        acier=CoucheConductriceCylindrique(50.2, 7.85e3, 1000, 0,
                                           0.05, 50.0)
        cylindre = Multicouche([acier])
        cylindre.definir_CL("D", "Convection", 200.0) # cavité d'eau
        acier2 = CoucheConductriceCylindrique(50.2, 7.85e3, 1000, 0.1,
                                              0.12, 10.0)
        coque = Multicouche([acier2])
        coque.definir_CL("G", "Convection", 200.0) # cavité d'eau
        coque.definir_CL("D", "Convection", 20.0)  # air extérieur
        coque.definir_signal("D", lambda s: 20.0/s)
        R_cav_min,R_cav_max = cylindre.X[-1],coque.X[0]
        V_sur_L = np.pi*(R_cav_max**2+R_cav_min**2)
        Smin_sur_L, Smax_sur_L = 2*np.pi*R_cav_min,2*np.pi*R_cav_max
        cavite = Cavite(V_sur_L, 1000.0, 4200.0,
                        [(cylindre,"D",Smin_sur_L),
                         (coque,"G",Smax_sur_L)], 10.0)
        essai5 = SystemeThermiqueLineaire(6.5*3600, 5.0, cavite,
                                          calculer=True)
        Vt,T_ext,F_ext = cylindre.T_phi(0.05)
        _,T_mil,F_mil = cylindre.T_phi(0)
        _,T_coq,F_coq = coque.T_phi(coque.X[-1])
        T_eau = essai5.T_cavites[0]
        fig = plt.figure("Températures et flux", figsize=(10,6))
        ax_temp,ax_flux = fig.subplots(2,1)
        fig.subplots_adjust(0.08, 0.09, 0.98, 0.99)
        ax_flux.set_xlabel(r"Instant $t\;[h]$")
        Vh = Vt/3600
        ax_temp.plot(Vh, T_eau, "-b", label=r"$T_{\mathrm{eau}}(t)$")
        ax_temp.plot(Vh, T_ext, "-g", label=r"$T_{\mathrm{ext}}(t)$")
        ax_temp.plot(Vh, T_mil, "-m", label=r"$T_{\mathrm{mil}}(t)$")
        ax_temp.plot(Vh, T_coq, "-r", label=r"$T_{\mathrm{coq}}$")
        ax_temp.set_ylabel(r"Température [°C]")
        ax_temp.legend(loc="best",fontsize=10) ; ax_temp.grid()
        ax_flux.plot(Vh, F_ext, "--b",
                     label=r"$\phi_{\mathrm{ext}}(t)$")
        ax_flux.plot(Vh, F_coq, "--r" ,
                     label=r"$\phi_{\mathrm{coq}}(t)$")
        ax_flux.set_ylabel(r"D.S. Flux [W/m²]")
        ax_flux.legend(loc="best", fontsize=10) ; ax_flux.grid()
        plt.show()
    #-------------------------------------------------------------------
