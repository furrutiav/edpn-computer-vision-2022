# Segmentación y pintado de imágenes

Análisis Numérico de Ecuaciones en Derivadas Parciales: Teoría y Laboratorio

Integrantes: 
* Daniel Minaya, 
* Sebastián Toloza, 
* Felipe Urrutia

## Propuesta proyecto

Resolver dos problemas distintos de EDP en procesamiento de imagenes, focalizando nuestro trabajo en el primero. (1) Segmentacion de imagenes (Image segmentation) y (2) Pintado de imagenes (Inpainting). La idea principal es utilizar (1) para encontrar un elemento de alto constraste sobre la imagen para luego aplicar (2) con la intencion de borrar dicho elemento preservando la informacion del borde. Para resolver (1) consideramos el metodo Level-set propuesto en [4, 2]. Mientras que para resolver (2) consideramos un metodo sencillo utilizando la ecuacion de Laplace [3]. Si durante el transcuso del proyecto logramos resolver satisfactoriamente el problema (1) y podemos mejorar el problema (2), entonces consideraremos una mejor aproximacion (2) con el modelo propuesto en [1].

**Figura 1.** *Esquema del procesamiento de una imagen*

<img src="https://github.com/furrutiav/edpn-computer-vision-2022/blob/main/Esquema.png" alt="drawing" width="650"/>

## Segmentacion de imagenes

[Notebook](https://github.com/furrutiav/edpn-computer-vision-2022/blob/main/01%20Level-set%20Image%20segmentation.ipynb)

## Pintado de imagenes
[Notebook]()

## Referencias

[[1]](https://dl.acm.org/doi/abs/10.1145/344779.344972)
Bertalmio, M., Sapiro, G., Caselles, V., & Ballester, C. (2000, July). Image inpainting. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques (pp. 417-424)

[[2]](https://publikationen.sulb.uni-saarland.de/bitstream/20.500.11880/26267/1/preprint_61_02.pdf)
Weickert, J., & Kühne, G. (2003). Fast methods for implicit active contour models. In Geometric level set methods in imaging, vision, and graphics (pp. 43-57). Springer, New York, NY.

[[3]](http://journals.du.ac.in/ugresearch/pdf-vol3/U13.pdf)
Agrawal, N., Sinha, P., Kumar, A., & Bagai, S. (2015). Fast & dynamic image restoration using Laplace equation based image inpainting. J Undergraduate Res Innovation, 1(2), 115-123.

[[4]](https://link.springer.com/content/pdf/10.1007/978-3-540-33267-1_12.pdf)
Frick, K., & Scherzer, O. (2007). Application of non-convex bv regularization for image segmentation. In Image Processing Based on Partial Differential Equations (pp. 211-228). Springer, Berlin, Heidelberg.

