# Segmentación de imágenes: Método del conjunto de nivel

**Curso:** *MA5307-1 Análisis Numérico de Ecuaciones en Derivadas Parciales: Teoría y Laboratorio*

**Institución:** *Departamento de Ingeniería Matemática, Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile*

**Autores:**
* Felipe Urrutia 
* Daniel Minaya 
* Sebastián Toloza

## Motivación

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

## Diferencias Finitas
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

## Imágenes Básicas

[imagenes pendientes]

## Métricas


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

[[1]](https://dl.acm.org/doi/abs/10.1145/344779.344972)
Bertalmio, M., Sapiro, G., Caselles, V., & Ballester, C. (2000, July). Image inpainting. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques (pp. 417-424)

[[2]](https://link.springer.com/chapter/10.1007/0-387-21810-6_3)
Weickert, J., & Kühne, G. (2003). Fast methods for implicit active contour models. In Geometric level set methods in imaging, vision, and graphics (pp. 43-57). Springer, New York, NY.

[[3]](https://www.researchgate.net/publication/311103980_Fast_Dynamic_Image_Restoration_using_Laplace_equation_Based_Image_Inpainting)
Agrawal, N., Sinha, P., Kumar, A., & Bagai, S. (2015). Fast & dynamic image restoration using Laplace equation based image inpainting. J Undergraduate Res Innovation, 1(2), 115-123.

[[4]](https://link.springer.com/chapter/10.1007/978-3-540-33267-1_12?noAccess=true)
Frick, K., & Scherzer, O. (2007). Application of non-convex bv regularization for image segmentation. In Image Processing Based on Partial Differential Equations (pp. 211-228). Springer, Berlin, Heidelberg.

