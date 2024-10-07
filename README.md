
# <span style="color:#0088BB"> <b>SysTherLin</b> </span>

Le code  <i>opensource</i> <b>SysTherLin</b> (pour « <u>Sys</u>tèmes <u>Ther</u>miques <u>Lin</u>éaires », Licence <tt>EUPL-1.2</tt>) est un code à vocation pédagogique développé par <b>Éric Ducasse</b> à l'<b>École Nat. Sup. d'Arts et Métiers (ENSAM)</b> de Bordeaux depuis 2018.<br>
Ce code est entièrement écrit en <b>Python 3</b>.<br>
L'objectif de ce code est de :

## <span style="color:#008800"> <b>Simuler le comportement au cours du temps d'un système thermique comportant des cavités et des multicouches 1D conducteurs.</b> </span>

Grâce à la linéarité du système, le calcul, exact, est fait dans le domaine de Laplace. Le passage, direct et inverse, vers le domaine temporel se fait par Transformée de Fourier Rapide (FFT). Voir par exemple :

- <b>J. W. Cooley, J. W. Tukey</b> (1965), <i>An algorithm for the machine calculation of complex Fourier series</i>, Math. Comput. <b>19</b>, 297-301 [Doi 10.1090/S0025-5718-1965-0178586-1](https://dx.doi.org/10.1090/S0025-5718-1965-0178586-1).
- <b>R.A. Phinney</b> (1965) <i>Theoretical calculation of the spectrum of first arrivals in layered elastic mediums</i>, J. Geophys. Res. <b>70</b>, 5107-5123 [Doi 10.1029/JZ070i020p05107](https://dx.doi.org/10.1029/JZ070i020p05107).
<!-- end of the list -->

### <span style="color:#0000FF"> <b>Modélisation d'un muticouche 1D conducteur</b></span>

1. L'état d'une couche conductrice (supposée homogène et sans source 
thermique à l'intérieur) est caractérisé par les champs de température 
$`T(r,t)`$ et de densité surfacique de flux de puissance $`\phi(r,t)`$, 
orienté selon les positions $`r`$ croissantes, vérifiant les équations 
aux dérivées  partielles linéaires suivantes :  
    ```math
    \left\lbrace\begin{array}{llll} 
    \displaystyle\partial_{t}T(r,t) & = & 
    \displaystyle \frac{-1}{\rho\,c_p}\,\left(\partial_{r}\phi(r,t) 
    + \frac{\boldsymbol{m}}{r}\;\phi(r,t)\right) & 
    \text{(conservation de l'énergie)}\\[2mm] 
    \phi(r,t) & = & \displaystyle-k\,\partial_{r}T(r,t) & 
    \text{(conduction)}\end{array}\right.
    ```
    avec les notations suivantes : 
    - $`k`$ la conductivité thermique [J/s/K/m]
    - $`\rho`$ la masse volumique [kg/m³]
    - $`c_p`$ la capacité thermique massique [J/K/kg]
    - $`\boldsymbol{m}`$ indicateur de géométrie : 
    $$`\displaystyle\left|\begin{array}{ll} \boldsymbol{m}=0\text{ :} & 
    \text{couche plane}\\ \boldsymbol{m}=1\text{ :} & 
    \text{couche à symétrie cylindrique}\\ 
    \boldsymbol{m}=2\text{ :} & \text{couche à symétrie sphérique} 
    \end{array}\right.`$$

1. À l'interface entre deux couches conductrices (nécessairement de 
même nature géométrique), on suppose qu'il y a continuité de la 
température et du flux (pas de résistance thermique de contact).

1. Les conditions limites au bord d'un multicouche peuvent être de 3 sortes :

    - Température imposée (type <i>Dirichlet</i>, à l'aide d'un thermostat, 
    difficile à réaliser)
    - Densité surfacique de flux imposée (type <i>Neumann</i>)
    - Convection (vers un gaz ou un fluide « agité »): la densité surfacique
     de flux de puissance thermique est proportionnelle à la différence de 
     températures entre la paroi et le fluide :<br> 
     $`\displaystyle\hspace{5mm}\phi(r,t) =
     \pm \eta\;\left(T(r,t)-T_{\text{fluide}}(t)\right)`$<br>
     où $`\eta`$ désigne le <i>coefficient de convection</i> [J/s/K/m²].
    - Dans le cas particulier d'un multicouche cylindrique ou sphérique 
    plein ($`r=0`$ sur la frontière gauche du multicouche), 
    il faut rajouter une <i>condition de symétrie</i>,  ce qui est 
    fait automatiquement dans le code.
<!-- end of the list -->

### <span style="color:#0000FF"> <b>Modélisation d'une cavité</b></span>

Une cavité (supposée contenir un gaz ou un fluide « agité ») est 
caractérisée par :

- son volume $`\mathcal{V}`$ [m³]
- sa masse volumique $`\rho`$ [kg/m³]
- sa capacité calorifique massique $`c_p`$ [J/K/kg]
- ses parois, chacune étant le bord d'un multicouche de surface 
$`\mathcal{S}_i`$ avec une <u>condition de convection vers la cavité</u> de 
coefficient $`\eta_i`$

L'équation de conservation d'énergie dans chaque cavité s'écrit :
```math
T_{\text{cavité}}^{\,\prime}(t) = 
\frac{1}{\rho\,c_p\,\mathcal{V}}\;
\sum_{i}\,\mathcal{S}_i\,\phi_{i}(t)\quad\text{avec}\quad 
\phi_{i}(t)=\eta_{i}\left(T_i(t)-T_{\text{cavité}}(t)\right)\,\text{,}
```
la température $`T_i(t)`$ de la paroi $`i`$ étant supposée uniforme.

### <span style="color:#0000FF"> <b>Utilisation du code</b></span>

Voir la documentation et les exemples fournis (programmes <i>python</i> et notebooks <i>ipython</i>).
