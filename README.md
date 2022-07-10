# Segmentación de imágenes: Método del conjunto de nivel

**Curso:** *MA5307-1 Análisis Numérico de Ecuaciones en Derivadas Parciales: Teoría y Laboratorio*

**Institución:** *Departamento de Ingeniería Matemática, Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile*

**Autores:**
* Felipe Urrutia 
* Daniel Minaya 
* Sebastián Toloza

## Motivación [[3]](https://github.com/furrutiav/edpn-computer-vision-2022/main/README.md#referencias)

La segmentación de imágenes es un método utilizado para particionar una imagen en múltiples segmentos u objetos constituyentes, lo cual la hace un componente esencial en muchos sistemas de comprensión visual, tales como, por ejemplo, el análisis de imágenes médicas, imágenes satelitales, entre otros.  

En este proyecto se buscó aplicar el método de segmentación de imágenes en niveles de gris a través de una EDP, haciendo uso del método de conjuntos de nivel.

## Método del Conjunto de Nivel

Se considera una interfaz descrita por una curva simple cerrada $\Gamma$, la 
cual separa un dominio $\Omega \subseteq \mathbb{R}^2$ en 
dos sub-dominios distintos de áreas no nulas, $\Omega^+,\Omega^-$, de 
fronteras respectivas $\partial \Omega^+, \partial \Omega ^-$, tales 
que:

$$ \Omega^+ \cup \Omega^- \cup \Gamma = \Omega $$

$$ \Omega^+ \cap \Omega^- = \emptyset $$

$$ \partial \Omega^+ \cap \partial \Omega ^- = \Gamma $$

Una forma práctica de describir tanto la interfaz como los sub-dominios es mediante la definición de una función implícita $u(x)$ tal que: 

$$ \Omega^+ = \lbrace{x \in \mathbb{R}^2 \mid u(x)>0\rbrace} $$

$$ \Omega^- = \lbrace{x \in \mathbb{R}^2 \mid u(x)<0\rbrace} $$

$$ \Gamma = \lbrace{x \in \mathbb{R}^2 \mid u(x)=0\rbrace} $$

[figura 1 pendiente]

## Ecuación en Derivadas Parciales: Modelo Geométrico

Dada una curva inicial $C_0$, la 
ecuación que se busca resolver para $u(x,t)$ es:

$$
        (\text{EDP})\quad 
        \begin{cases}
        \frac{\partial u}{\partial t} =g(x)|\nabla u| \left( \text{div}\left( \frac{\nabla u}{|\nabla u|} \right)+\kappa \right) & \text{en } \Omega \times (0,\infty) \\
        u(x,0) = u_0(x) & \text{en } \Omega
        \end{cases}
$$

donde $u_0(x)$ es 
una función distancia con signo, dada por:

$$
        u_0(x)=
        \begin{cases}
        d(x,C_0) & \text{si $x$ está \textbf{dentro} de $C_0$}\\
        0 & \text{si $x$ está \textbf{en} $C_0$}\\
        -d(x,C_0)  & \text{si $x$ está \textbf{fuera} de $C_0$}
        \end{cases}
$$

Tenemos que $g$ es una *stopping function*, dada por:

$$
        g(x) = \frac{1}{1+|\nabla f_{\sigma}(x)|^2/\lambda^2},
$$
  
donde $f_{\sigma}$ corresponde 
a la suavización de la imagen a partir de un kernel gaussiano de desviación estándar $\sigma$ y 
$\lambda$ es 
un factor de contraste, $\kappa$ es 
un término constante de fuerza comparable a la fuerza de un globo y el término $\text{div}\left( \frac{\nabla u}{|\nabla u|} \right)$ hace 
referencia a la curvatura media de la interfaz.

## Diferencias Finitas [[4]](https://github.com/furrutiav/edpn-computer-vision-2022/main/README.md#referencias)

La implementación en este caso viene descrita por la relación:

$$
        \left( I - \tau A(u^n)\right)u^{n+1} = u^n + \kappa\tau|\nabla^-u|^n g,
$$

donde los coeficientes de $A$ vienen 
dados por:

$$
        A_{ij}(u^n):=
        \begin{cases}
        g_i|\nabla u|_i^n\left(\frac{2}{{|\nabla u|} _i^n+{|\nabla u|}_j^n}\right) & j \in N(i) \\
        -g_i|\nabla u|_i^n \sum_{m \in N(i)} \left( \frac{2}{|\nabla u|_i^n+|\nabla u|_m^n} \right) & j=i\\
        0 & \text{otro caso}
        \end{cases}
$$

Las aproximaciones de $|\nabla u|$ vienen 
dadas según los siguientes casos:
1. Si $\kappa \leq 0$, entonces 
$|\nabla u|_i^n     \approx  |\nabla^- u|_i^n $, que 
viene dado por:

$$
            |\nabla^- u|_i^n = 
            (\max(D^{-x}u_i^n,0)^2 + \min(D^{+x}u_i^n,0)^2 
             + \max(D^{-y}u_i^n,0)^2 + \min(D^{+y}u_i^n,0)^2     )^{1/2}   
$$

2. Si $\kappa > 0$, entonces 
$|\nabla u|_i^n     \approx  |\nabla^+ u|_i^n $, que 
viene dado por:

$$
            |\nabla^+ u|_i^n =
            (\min(D^{-x}u_i^n,0)^2 + \max(D^{+x}u_i^n,0)^2  + \min(D^{-y}u_i^n,0)^2 + \max(D^{+y}u_i^n,0)^2     )^{1/2}
$$

## Imágenes Básicas [[1]](https://github.com/furrutiav/edpn-computer-vision-2022/main/README.md#referencias)

[imagenes pendientes]

## Métricas [[2]](https://github.com/furrutiav/edpn-computer-vision-2022/main/README.md#referencias)


$$ 
    \texttt{Precision}(\Omega_\text{EDP}, +) = \frac{|\Omega_\text{EDP}^+ \cap \Omega_\text{target}^+|}{|\Omega^+_\text{EDP}|}
$$

$$
    \texttt{Recall}(\Omega_\text{EDP}, +) = \frac{|\Omega_\text{EDP}^+ \cap \Omega_\text{target}^+|}{|\Omega_\text{target}^+|}
$$

$$
    \texttt{F1-score}(\Omega_\text{EDP}, +) = \textit{HM}(\texttt{Precision}(\Omega_\text{EDP}, +),\texttt{Recall}(\Omega_\text{EDP}, +))
$$

donde *HM* corresponde 
al promedio armónico.
    
## Imágenes Reales

[imagenes pendientes]

## Referencias

[1] M. M. Bongard. Pattern recognition. Rochelle Park, N.J.: Hayden Book Co., Spartan Books. (Original publication: Nauka Press, Moscow), 1967

[2] M. Everingham, L. VanGool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results.

[3] S. Minaee, Y. Y. Boykov, F. Porikli, A. J. Plaza, N. Kehtarnavaz, and D. Terzopoulos. Image segmentation using deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence, 2021

[4] J. Weickert and G. Kühne. Fast methods for implicit active contour models. In Geometric level set methods in imaging, vision, and graphics. Springer, New York, NY, pages 43–57, 2003.

## Citar
```
@software{Urrutia_edpn-computer-vision-2022_2022,
author = {Urrutia, Felipe and Minaya, Daniel and Toloza, Sebastian},
doi = {10.5281/zenodo.1234},
month = {6},
title = {{edpn-computer-vision-2022}},
url = {https://github.com/furrutiav/edpn-computer-vision-2022},
version = {1.0.0},
year = {2022}
}
```

