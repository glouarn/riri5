from scipy import *
import riri5 as riri

import RIRI5 as riri #celui du dossier l-egume : qui marche


#test 1 :
m = array([[0.,1.,3.,3.,1.,0.],[0.,1.,3.,3.,1.,0.], [0.,1.,3.,3.,1.,0.], [0.,1.,3.,3.,1.,0.]])
ma = array([m, m,m, m])/4.
ma2 = array([m, m,m, m])/8.
ls_mlai = array([ma, ma2])


ves = array([ma[:,2,2], ma[:,1,1]]) #liste de lai par especeet voxel sur le chemin du rayon
ks = array([1., 0.4]) #liste de k par espece
I0 = 1000.

#pour 1 rayon
res_trans, res_abs_i = riri.calc_extinc_ray_multi(ves, I0, ks) # proche d'un lai de 4 : 1000.*exp(-1.*4.)


#calcul des listes de triplets selon la matrice
triplets = riri.get_ls_triplets(ma, opt='V')
triplets = riri.get_ls_triplets(ma, opt='VXpXmYpYm')


#distributtions d'angles
ls_distf = [riri.disttetaf(abs(45.), 0.), riri.disttetaf(abs(45.), 0.)]

#pour tous les rayons
res_trans_form, res_abs_i_form = riri.calc_extinc_allray_multi(ls_mlai, triplets ,ls_distf , I0)




#test 2: calcul sur matrice creuse avec toutes les cases
ma3 = zeros([8,4,4])
ma3[6:8,:,:] = 1.
ma4 = zeros([8,4,4])
ma4[5:8,:,:] = 0.5
ls_mlai2 = array([ma3, ma4])


triplets = riri.get_ls_triplets(ma3, opt='VXpXmYpYm')
ls_distf = [riri.disttetaf(abs(45.), 0.), riri.disttetaf(abs(45.), 0.)]
res_trans_form, res_abs_i_form = riri.calc_extinc_allray_multi(ls_mlai2, triplets ,ls_distf , I0)


#test 3: avec fonction qui redimensionne (plus rapide!)

res_trans_form, res_abs_i_form = riri.calc_extinc_allray_multi_reduced(ls_mlai2, triplets ,ls_distf , I0, optsky=None, opt='VXpXmYpYm')
