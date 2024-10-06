# Version 1.30 - 2024, October, 6
# Project : SysTherLin (Systèmes thermiques linéaires)
# Copyright (Eric Ducasse 2018)
# Licensed under the EUPL-1.2 or later
# Institution : ENSAM / I2M
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########################################################################
## EXEMPLE N°2 : TUBE CONTENANT DE L'EAU DANS UNE CAVITÉ SPHÉRIQUE #####
########################################################################
import numpy as np
import matplotlib.pyplot as plt
# Pour aller chercher les modules de SysTherLin :
import os,sys
rel_sys_ther_lin_path = ".." # Chemin relatif du dossier SysTherLin
abs_sys_ther_lin_path = os.path.abspath(rel_sys_ther_lin_path)
sys.path.append(abs_sys_ther_lin_path)
# ---
from Couches_conductrices import (CoucheConductriceCylindrique,
                                  CoucheConductriceSpherique,
                                  CoucheConductrice,Multicouche)
from Systemes_thermiques_lineaires import (Cavite,
                                           SystemeThermiqueLineaire)
## 1 - Définition des couches conductrices
T0 = 60.0 # Température initiale
## 1.1 - Couches sphériques de la cloche
inox1 = CoucheConductriceSpherique(16.5, 8000.0, 500.0,
                                   0.150, 0.152, T0)
pstyr = CoucheConductriceSpherique(0.04, 18.0, 1450.0,
                                   0.152, 0.157, T0)
inox2 = CoucheConductriceSpherique(16.5, 8000.0, 500.0,
                                   0.157, 0.159, T0)
## 1.2 - Couches cylindriques du tube
# Polypropylène avec 25% de fibres de verre.
pp1 = CoucheConductriceCylindrique(0.22, 910.0, 1800.0,
                                   25.0e-3, 26.5e-3, T0)
air = CoucheConductriceCylindrique(0.026, 1.2, 1000.0,
                                   26.5e-3, 28.5e-3, T0)
pp2 = CoucheConductriceCylindrique(0.22, 910.0, 1800.0,
                                   28.5e-3, 30.0e-3, T0)
## 1.3 - Couches planes du socle
inox_socle_inf = CoucheConductrice(16.5, 8000.0, 500.0, 2e-3, T0)
polystyrene_socle = CoucheConductrice(0.04, 18.0, 1450.0, 10e-3, T0)
inox_socle_sup = CoucheConductrice(16.5, 8000.0, 500.0, 2e-3, T0)
## 2 - Définition des multicouches avec conditions limites
# 2.1 Cloche hémisphérique
coque = Multicouche([inox1, pstyr, inox2])
coque.definir_CL("G", "Convection", 10.0) # Côté intérieur
coque.definir_CL("D", "Convection", 10.0) # Côté extérieur
print(coque)
# 2.2 Tube cylindrique
tube = Multicouche([pp1, air, pp2])
tube.definir_CL("G", "Convection", 200.0) # Côté intérieur
tube.definir_CL("D", "Convection", 10.0) # Côté extérieur
print(tube)
# 2.3 Plateau plan
socle = Multicouche([inox_socle_inf, polystyrene_socle, inox_socle_sup])
socle.definir_CL("G", "Convection", 10.0) # Côté inférieur
socle.definir_CL("D", "Convection", 10.0) # Côté supérieur
print(socle)
## 3 - Définition des cavités
L_tube = 0.12
V_eau = np.pi * tube.X[0]**2 * L_tube
V_tube = np.pi * tube.X[-1]**2 * L_tube
V_air = 2/3*np.pi * coque.X[0]**3 - V_tube
eau = Cavite(V_tube, 1000.0, 4200.0,[
             [tube, "G", 2*np.pi*tube.X[0]*L_tube]], T0)
air = Cavite(V_air, 1.2, 1000.0,[ [coque, "G", 2*np.pi*coque.X[0]**2],
                                  [socle, "D", np.pi*(coque.X[0]**2-tube.X[-1]**2)],
                                  [tube, "D", 2*np.pi*tube.X[-1]*L_tube]
                                ], T0)
## 4 - Définition du systeme global
systeme = SystemeThermiqueLineaire(2*3600, 1.0, [air,eau])

## 5 - Calcul et visualisation des données
# Température extérieure de 10°C : transformée de Laplace 10/s
socle.definir_signal("G", lambda s : 10.0/s)
coque.definir_signal("D", lambda s : 10.0/s)
systeme.calculer_maintenant()
## 6 - Visualisation des résultats
fig = plt.figure("Résultats de la simulation", figsize=(14,7))
ax_T,ax_phi = fig.subplots(1,2)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.94,
                    wspace=0.3)
tmn = systeme.timeValues/60.0
### Températures ###
ax_T.set_title("Températures")
ax_T.set_xlabel("Instant t [minutes]")
# Température de l'eau
T_eau = systeme.T_cavites[1]
ax_T.plot(tmn, T_eau, "-", color=(0,0.5,0.5), linewidth=2.0,
          label = "T° eau")
# Températures dans le tube
_,T_tube_int,_ = tube.T_phi(0.5*(tube.X[0]+tube.X[1]))
ax_T.plot(tmn, T_tube_int, "-", color=(0.8,0,0.8), linewidth=2.0,
          label = "T° tube int.")
_,T_tube_ext,_ = tube.T_phi(0.5*(tube.X[2]+tube.X[3]))
ax_T.plot(tmn, T_tube_ext, "-", color=(0.8,0,0), linewidth=2.0,
          label = "T° tube ext.")
# Température de l'air
T_air = systeme.T_cavites[0]
ax_T.plot(tmn, T_air, "-", color=(0,0.5,0), linewidth=2.0,
          label = "T° air")
# Température sur la paroi supérieure du socle
_,T_socle_sup,_ = socle.T_phi(socle.X[-1])
ax_T.plot(tmn, T_socle_sup, "--", color=(0.6,0.6,0), linewidth=2.0,
          label = "T° plat. sup.")
# Température sur la paroi intérieure de la coque
_,T_coque_int,_ = coque.T_phi(coque.X[0])
ax_T.plot(tmn, T_coque_int, "--", color=(0.8,0.5,0), linewidth=2.0,
          label = "T° cl. int.")
# Température dans le socle (milieu du polystyrène)
_,T_socle,_ = socle.T_phi(0.5*(socle.X[0]+socle.X[-1]))
ax_T.plot(tmn, T_socle, "-", color=(0.6,0.6,0), linewidth=2.0,
          label = "T° plateau")
# Température dans la coque (milieu du polystyrène)
_,T_coque,_ = coque.T_phi(0.5*(coque.X[0]+coque.X[-1]))
ax_T.plot(tmn, T_coque, "-", color=(0.8,0.5,0), linewidth=2.0,
          label = "T° cloche")
# Température sur la paroi inférieure du socle
_,T_socle_inf,_ = socle.T_phi(socle.X[0])
ax_T.plot(tmn, T_socle_inf, ":", color=(0.6,0.6,0), linewidth=2.0,
          label = "T° plat. inf.")
# Température sur la paroi extérieure de la coque
_,T_coque_ext,_ = coque.T_phi(coque.X[-1])
ax_T.plot(tmn, T_coque_ext, ":", color=(0.8,0.5,0), linewidth=2.0,
          label = "T° cl. ext.")
# Finalisation du tracé
ax_T.set_ylabel("Températures [°C]")
ax_T.grid() ; ax_T.legend(loc="best", fontsize=10)
### Flux ###
ax_phi.set_xlabel("Instant t [minutes]")
ax_phi.set_title("Densités surfaciques de Flux")
_,_,phi_socle_sup = socle.T_phi(socle.X[-1])
ax_phi.plot(tmn, phi_socle_sup, "-", color=(0.6,0.6,0), linewidth=2.0,
            label = r"$\phi$ plateau sup")
_,_,phi_socle_inf = socle.T_phi(socle.X[0])
ax_phi.plot(tmn, phi_socle_inf, "--", color=(0.6,0.6,0),
            linewidth=2.0, label = r"$\phi$ plateau inf")
_,_,phi_coque_int = coque.T_phi(coque.X[0])
ax_phi.plot(tmn, -phi_coque_int, "-", color=(0.8,0.5,0),
            linewidth=2.0, label = r"$-\phi$ cloche int")
_,_,phi_coque_ext = coque.T_phi(coque.X[-1])
ax_phi.plot(tmn, -phi_coque_ext, "--", color=(0.8,0.5,0),
            linewidth=2.0, label = r"$-\phi$ cloche ext")
_,_,phi_tube = tube.T_phi(tube.X[-1])
ax_phi.plot(tmn, phi_tube, "-", color=(0.8,0,0), linewidth=2.0,
            label = r"$\phi$ tube")
ax_phi.grid() ; ax_phi.legend(loc="best", fontsize=10)
ax_phi.set_ylabel(r"$\phi(t)$ [W/m²/K]")
plt.show()
