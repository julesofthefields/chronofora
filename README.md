# Chronofora: How Google Maps sees the world

## Introduction

Planet Earth is commonly represented as a sphere. This truism is by no means intended as an insult to flat-earthers. Neither does it aim at reducing planet Earth to its surface only – indeed, a more correct word, inclusive of the inside of Planet Earth, could have been “ball”, but after all humans have almost exclusively known the surface of the Earth only: the deepest point can be found in the Mariana Trench in the Western Pacific Ocean, at ~11km deep, and the highest point Mount Everest stands at ~9km tall, making it that humans have throughout history spent their entire lives at 6371km from the center of the Earth, plus or minus 0.15%. So, yes, let us represent the Earth as a sphere. A globe.

This representation is accurate visually, of course. A globe is an accurate model of planet Earth, as a geographical map. However, the purposes of a map can be multifold. For instance, the first maps that were drawn reflected a practical understanding of the space, rather than the exact layout of the environment. In effect, these two aspects could have been difficult to tell apart because the speed of transportation was quasi-constant, considering the lack of diversity in means of transportation. But, this practical understanding of the space is essential in constructing a mental model of transportation within the environment. As an example, Tabula Peutingeriana, a Roman "road map" from the 4th or 5th century CE, doesn't represent geography accurately — it lays out roads like a subway map, focusing on stops and travel stages, not scale. 


This project provides a framework to deform 2D surfaces in 3D according to custom metrics. In other words, it reshapes the geometry of a given closed 2D surface in 3D so that distances between any 2 points reflect a custom distance. In more pompous words, it tries to construct a quasi-isometric 2D embedding in a 3D space, of a 2D surface in 3D, seen under a custom metric.

In this particular case, I wanted to deform the Earth, represented by the unit sphere in 3D, so that distances on the deformed surfaces would represent _at best_, in some sense of the terms, the travel times between 2 points. Chronofora is the name I have given to this new Earth-potato (but feel free to let me know if you have any better ideas).

The aim of this project is mainly artistic I guess. But beyond providing some cool and fun shapes, it also stands as an algorithm to visualize how given metrics can distort 2D surfaces in 3D. And as far as I know, there are no other algorithms in the literature which seem to tackle this visualization problem, as all classical visualization methods assume a flat Euclidean target space.


