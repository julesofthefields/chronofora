# Chronofora: How Google Maps sees the world

## Introduction

This project provides a framework to reshape the geometry of a given closed 2D surface in 3D so that distances between any 2 points reflect a custom distance. In more pompous words, it tries to construct a quasi-isometric 2D embedding in a 3D space, of a 2D surface in 3D, seen under a custom metric.

In this particular case, I wanted to deform the Earth, represented by the unit sphere in 3D, so that distances on the deformed surfaces would represent _at best_, in some sense of the terms, the travel times between 2 points. To give an idea, from New York it takes 6-7h to get to Paris, while it takes 2-3 days to get to the middle of the Atlantic Ocean, in terms of fastest accessible transportation. So on the deformed version of the Earth, the geodesic distance (meaning, the shortest distance on the surface) between what are supposed to be New York and Paris is much smaller than the distance to the middle of the Atlantic Ocean. As you might have guessed, Chronofora is the tentative name I have given to this new Earth-potato planet (feel free to let me know if you have any better ideas).

You might also have guessed that this problem does not seem to have a solution: we can  safely assume that on the Earth, all the points lying on the mediatrix between NY and Paris are further away from either of them than the other city. With this in mind, if there existed an accurate 3-dimensional deformation of the Earth, it would necessarily violate continuity, and therefore not be a deformation as intended. This is one of the reasons why the embedding constructed is _quasi_-isometric.

Originally, the aim of this project is mainly artistic. But beyond providing some cool and fun shapes, it also stands as an algorithm to visualize how given metrics can distort 2D surfaces in 3D, which may prove useful in some more concrete cases. And as far as I know, there are no other algorithms in the literature which seem to tackle this visualization problem, as all classical visualization methods assume a flat Euclidean target space.

This is only a first version of Chronofora at this stage (there is a lot of room for improvement), but please let me know what you think!

## Features and Project Structure

The main part of the project resides in the file __get_mesh.py__. That's where it loads data associated to Chronofora, asks which cities you might want to see plotted on it, and then pops up a few windows for visualization. Feel free to check what the geometry looks like, and know there might seem to be some incoherences: this problem does not have a clear solution, and there have been some arbitrary choices in the parameters in favor of the "coolness" of the shape of Chronofora. So it is likely that there are some  pairs of points between which transportation time is very long, but which seem to weirdly close on Chronofora, or even vice-versa. This is subject to improvmeent.

The rest of the project is structured as such:
- __get_X_transportation_geometry.py__ processes open-source transportation datasets (flight, train, ferry, car) to compute transportation times between fixed "transportation stations";
- __get_X_geometry.py__ proposes a heuristic method to compute multimodal travel times between arbitrary global locations, and then computes the pairwise distances of a regular sample of N points on Planet Earth;
- __surface_spring_system.py__ builds the object necessary to deform the surface -- a large-scale spring system, with some additional constraints to keep the 2D aspect of the surface -- and __run_spring.py__ runs the simulation;
- __map_utils.py__ provides some utiliary functions.

## More technical paper

To follow.

## Acknowledgments

The idea for this project actually originated a few years back, when I was interning at GeomStats under the supervision of Nina Miolane (UC Santa Barbara) and Alice Le Brigant (Université Paris Panthéon Sorbonne). Professor Andrew Blumberg (Columbia University) also helped me in approaching the problem. 


