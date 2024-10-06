# Version 1.30 - 2024, October, 6
# Project : SysTherLin (Systèmes thermiques linéaires)
# Copyright (Eric Ducasse 2018)
# Licensed under the EUPL-1.2 or later
# Institution : ENSAM / I2M
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########################################################################
#####    EXEMPLE ÉLÉMENTAIRE D'UN BICOUCHE    ##########################
########################################################################
import numpy as np
import matplotlib.pyplot as plt
# Pour aller chercher les modules de SysTherLin :
import os,sys
rel_sys_ther_lin_path = ".." # Chemin relatif du dossier SysTherLin
abs_sys_ther_lin_path = os.path.abspath(rel_sys_ther_lin_path)
sys.path.append(abs_sys_ther_lin_path)
# ---
from Couches_conductrices import CoucheConductrice,Multicouche
from Systemes_thermiques_lineaires import (Cavite,
                                           SystemeThermiqueLineaire)
## 1 - Définition des couches conductrices
# 3mm d'épaisseur et température initiale de 20°C
inox = CoucheConductrice(16.5, 8000.0, 500.0, 3.0e-3, 20.0)
# 10cm d'épaisseur et température initiale de 20°C
verre =  CoucheConductrice(5.0, 2800.0, 1000.0, 0.1, 20.0) 
## 2 - Définition du multicouche avec conditions limites
bicouche = Multicouche([inox, verre])
bicouche.definir_CL("G", "Neumann") # Côté chauffage
bicouche.definir_CL("D", "Convection", 10.0) # Côté air extérieur
## ( 3 - Définition de la cavité )
# Sans objet ici
## 4 - Définition du systeme global
jour = 3600.*24
STL = SystemeThermiqueLineaire(4.2*jour, 10.0, bicouche)
## 5 - Calcul et visualisation des données
# 5.1 Définition du signal de chauffage
def chauf(t) :
    h = (t/3600.0)%24.0
    return ((h>23)|(h<7))*100.0 # 100 W/m² s'il est allumé
# On récupère tous les instants de calcul
instants_positifs = STL.positiveTimeValues
# On définit le vecteur représentant les valeurs du signal échantillonné
valeurs_chauffage = chauf(instants_positifs)
# Lorsque le deuxème argument de la méthode definir_signal est un
# vecteur, cela signifie implicitement qu'il s'agit des valeurs du
# signal échantillonné
bicouche.definir_signal("G", valeurs_chauffage)
# 5.2 Définition de la température extérieure
def T_ext(t) :
    amplitude = 10
    w = 2*np.pi/(24*3600) # Période de 24h
    tau = 16*3600 # Retard pour que le maximum soit à 16h
                  #                    et le minimum à 4h
    return 15.0 + amplitude*np.cos( w*(t-tau) )
valeurs_temp_exte = T_ext(instants_positifs)
bicouche.definir_signal("D", valeurs_temp_exte)
# 5.3 Calcul de la solution
STL.calculer_maintenant()

## 6 - Visualisation des résultats
instants = STL.timeValues    # Les instants d'échantillonnage sont
t_en_jours = instants / jour # communs à tous les signaux
# Tous les tracés sur la même figure : plusieurs systèmes d'axes                             
fig = plt.figure("Graphiques", figsize=(12,7) )
ax_input, ax_output = fig.subplots(2,1)
ax_input2 = ax_input.twinx() # 2ème système d'axes à droite
plt.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.93, \
                    wspace=0.4, hspace=0.3)
ax_input.set_title("Signaux d'entrée", family="Arial", size=14 )
ax_output.set_xlabel("Instant t [jour]")
ax_input.set_ylabel("Puissance fournie [W/m²]", color=(0.6,0,0))
ax_input2.set_ylabel("Température extérieure [°C]",color=(0,0,0.8))
ax_input.grid()

# Signaux d'entrée :
# Tracé du signal à gauche : dens. surf. de flux de puissance calo.
ax_input.plot(t_en_jours, bicouche.signal("G"), ".r", markersize=1.2)
# Tracé du signal à droite : température extérieure
ax_input2.plot(t_en_jours, bicouche.signal("D"), ".b", markersize=1.2)

# Résultats de la simulation:
ax_output.set_title("Résultats de la simulation", family="Arial",
                    size=14 )
# Température au coeur de l'inox du socle
z_mil_inox = 0.5*inox.e # position du milieu de l'inox
             # aussi 0.5*(bicouche.X[0]+bicouche.X[1])
_,T_inox,_ = bicouche.T_phi(z_mil_inox)
ax_output.plot(t_en_jours, T_inox, "-", color=(0.8,0,0),
               linewidth=2.0,
               label = f"T° inox, $x={1e3*z_mil_inox:.1f}$ mm")
# Température du verre du côté extérieur
Vx = np.array([0.8*bicouche.X[1]+0.2*bicouche.X[2],
               0.5*bicouche.X[1]+0.5*bicouche.X[2], bicouche.X[2]])
for x in Vx :
    _,T_verre,_ = bicouche.T_phi(x)
    ax_output.plot(t_en_jours, T_verre, "-", linewidth=2.0,
                   label = f"T° verre, $x={1e3*x:.1f}$ mm")
# Finalisation du tracé
ax_output.set_ylabel("Températures [°C]")
ax_output.grid()
ax_output.legend(loc="upper right", fontsize=10)
# Affichage à la fin
plt.show()
