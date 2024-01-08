#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python GetFEM interface
#
# Copyright (C) 2004-2020 Yves Renard, Julien Pommier.
#
# This file is a part of GetFEM
#
# GetFEM  is  free software;  you  can  redistribute  it  and/or modify it
# under  the  terms  of the  GNU  Lesser General Public License as published
# by  the  Free Software Foundation;  either version 2.1 of the License,  or
# (at your option) any later version.
# This program  is  distributed  in  the  hope  that it will be useful,  but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or  FITNESS  FOR  A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
# You  should  have received a copy of the GNU Lesser General Public License
# along  with  this program;  if not, write to the Free Software Foundation,
# Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301, USA.
#
############################################################################

import numpy as np

# Import basic modules
import getfem as gf

import scipy as sci

from io import StringIO
from scipy.io import mmread, mmwrite
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp

# from decimal import *
# getcontext().prec = 100 # Define precision: the maximum number of digits after the decimal point

## Parameters

Elastic_Modulus = np.array([5e3, 2e3, 2e2])
Poisson_ratio = np.array([0.25, 0.25, 0.4])
Lambda = Elastic_Modulus*Poisson_ratio/((1+Poisson_ratio)*(1-2*Poisson_ratio))
Mu =Elastic_Modulus/(2*(1+Poisson_ratio))

plane_xy = np.array([8,4])  # XY plane for multi-layer contact system
hight_z = np.array([0.4,0.8,1.6]) # The layer height of each layer
n_layer = hight_z.size # number of layers for multi-layer contact system

N_xy = np.array([25,16]) # [40,20] # Mesh parameters for XY plane
N_z = np.array([4,8,16]) # Mesh parameters for each layer

# f0 and f1 
pressure = 'Heaviside(X(1)-3.8)*Heaviside(4.4-X(1))*Heaviside(X(2)-1.8)*Heaviside(2.2-X(2))*[-4.5,0,-22.5]' # [-4.5,0,-22.5]
gravity_1 = '[0,0,-0.05]'
gravity_2 = '[0,0,-0.05]'
gravity_3 = '[0,0,-0.05]'
# friction function
fun_g_1 = lambda x: 0.2
fun_g_2 = lambda x: 0.05

degree = 1 # Order k of finite element polynomial space
degree_lambda = 0 # Definition of boundary element space

Dirichlet_Penalization_coefficient = 1e30

# Create a simple cartesian mesh
m_l_1 = gf.Mesh('regular_simplices', np.arange(0, plane_xy[0]+plane_xy[0]/N_xy[0], plane_xy[0]/N_xy[0]),
                np.arange(0, plane_xy[1]+plane_xy[1]/N_xy[1], plane_xy[1]/N_xy[1]), 
                np.arange(hight_z[2]+hight_z[1], round(hight_z[2]+hight_z[1]+hight_z[0]+hight_z[0]/N_z[0],5), round(hight_z[0]/N_z[0],5)))
m_l_2 = gf.Mesh('regular_simplices', np.arange(0, plane_xy[0]+plane_xy[0]/N_xy[0], plane_xy[0]/N_xy[0]),
                np.arange(0, plane_xy[1]+plane_xy[1]/N_xy[1], plane_xy[1]/N_xy[1]), 
                np.arange(hight_z[2], round(hight_z[2]+hight_z[1]+hight_z[1]/N_z[1],5), round(hight_z[1]/N_z[1],5)))
m_l_3 = gf.Mesh('regular_simplices', np.arange(0, plane_xy[0]+plane_xy[0]/N_xy[0], plane_xy[0]/N_xy[0]),
                np.arange(0, plane_xy[1]+plane_xy[1]/N_xy[1], plane_xy[1]/N_xy[1]), 
                np.arange(0, round(hight_z[2]+hight_z[2]/N_z[2],5), round(hight_z[2]/N_z[2],5)))

m_l_1.export_to_vtk('Mesh_1.vtk')
m_l_2.export_to_vtk('Mesh_2.vtk')
m_l_3.export_to_vtk('Mesh_3.vtk')

# Create a MeshFem for u and rhs fields of dimension 3
mfu_1 = gf.MeshFem(m_l_1,3) # displacement
mfp_1 = gf.MeshFem(m_l_1,1) # pressure
mfc_N1 = gf.MeshFem(m_l_1,1) # lambda_N
mfc_T1 = gf.MeshFem(m_l_1,3) # lambda_T
mfu_1.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree,))) # Assign the Pnk fem to all elements of the both MeshFem for layer 1
mfp_1.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,1)'))
mfc_N1.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree_lambda,)))
mfc_T1.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree_lambda,)))
mim_1=gf.MeshIm(m_l_1, gf.Integ('IM_TETRAHEDRON(5)')) # Integration method used

mfu_2 = gf.MeshFem(m_l_2,3) # displacement
mfp_2 = gf.MeshFem(m_l_2,1) # pressure
mfc_N2 = gf.MeshFem(m_l_2,1) # lambda_N
mfc_T2 = gf.MeshFem(m_l_2,3) # lambda_T
mfu_2.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree,))) # Assign the Pnk fem to all elements of the both MeshFem for layer 2
mfp_2.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,1)'))
mfc_N2.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree_lambda,)))
mfc_T2.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree_lambda,)))
mim_2=gf.MeshIm(m_l_2, gf.Integ('IM_TETRAHEDRON(5)')) # Integration method used

mfu_3 = gf.MeshFem(m_l_3,3) # displacement
mfp_3 = gf.MeshFem(m_l_3,1) # pressure
mfc_N3 = gf.MeshFem(m_l_3,1) # lambda_N
mfc_T3 = gf.MeshFem(m_l_3,3) # lambda_T
mfu_3.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree,))) # Assign the Pnk fem to all elements of the both MeshFem for layer 3
mfp_3.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,1)'))
mfc_N3.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree_lambda,)))
mfc_T3.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree_lambda,)))
mim_3=gf.MeshIm(m_l_3, gf.Integ('IM_TETRAHEDRON(5)')) # Integration method used

print('Number of convexes of Mesh for layer 1 = %d' % (m_l_1.nbcvs()))
print('Number of ponits of Mesh for layer 1 = %d' % (m_l_1.nbpts()))
print('The dimension of the field interpolated by the MeshFem for layer 1 = %d' % (mfu_1.qdim()))
print('FEM used by the MeshFem for layer 1 = %s' % (mfu_1.fem()[0].char()))
print('Number of degrees of freedom (dof) of the MeshFem for layer 1 = %d' % (mfu_1.nbdof()))
print()

print('Number of convexes of Mesh for layer 2 = %d' % (m_l_2.nbcvs()))
print('Number of ponits of Mesh for layer 2 = %d' % (m_l_2.nbpts()))
print('The dimension of the field interpolated by the MeshFem for layer 2 = %d' % (mfu_2.qdim()))
print('FEM used by the MeshFem for layer 2 = %s' % (mfu_2.fem()[0].char()))
print('Number of degrees of freedom (dof) of the MeshFem for layer 2 = %d' % (mfu_2.nbdof()))
print()

print('Number of convexes of Mesh for layer 3 = %d' % (m_l_3.nbcvs()))
print('Number of ponits of Mesh for layer 3 = %d' % (m_l_3.nbpts()))
print('The dimension of the field interpolated by the MeshFem for layer 3 = %d' % (mfu_3.qdim()))
print('FEM used by the MeshFem for layer 3 = %s' % (mfu_3.fem()[0].char()))
print('Number of degrees of freedom (dof) of the MeshFem for layer 3 = %d' % (mfu_3.nbdof()))
print()

# Boundary selection
bd_1 = m_l_1.outer_faces() # Boundary list of Layer 1
bd_norm_1 = m_l_1.normal_of_faces(bd_1) # Normal vector for Boundary of Layer 1
bd_11_tf = abs(bd_norm_1[2,:]) < 1e-14
bd_12_tf = abs(bd_norm_1[2,:]-1) < 1e-14
bd_13_tf = abs(bd_norm_1[2,:]+1) < 1e-14
bd_11 = np.compress(bd_11_tf, bd_1, axis=1)
bd_12 = np.compress(bd_12_tf, bd_1, axis=1)
bd_13 = np.compress(bd_13_tf, bd_1, axis=1)
Boundary_Region_11 = 11; Boundary_Region_12 = 12; Boundary_Region_13 = 13;
m_l_1.set_region(Boundary_Region_11, bd_11)
m_l_1.set_region(Boundary_Region_12, bd_12)
m_l_1.set_region(Boundary_Region_13, bd_13)

bd_2 = m_l_2.outer_faces() # Boundary list of Layer 2
bd_norm_2 = m_l_2.normal_of_faces(bd_2) # Normal vector for Boundary of Layer 2
bd_21_tf = abs(bd_norm_2[2,:]) < 1e-14
bd_22_tf = abs(bd_norm_2[2,:]-1) < 1e-14
bd_23_tf = abs(bd_norm_2[2,:]+1) < 1e-14
bd_21 = np.compress(bd_21_tf, bd_2, axis=1)
bd_22 = np.compress(bd_22_tf, bd_2, axis=1)
bd_23 = np.compress(bd_23_tf, bd_2, axis=1)
Boundary_Region_21 = 21; Boundary_Region_22 = 22; Boundary_Region_23 = 23;
m_l_2.set_region(Boundary_Region_21, bd_21)
m_l_2.set_region(Boundary_Region_22, bd_22)
m_l_2.set_region(Boundary_Region_23, bd_23)

bd_3 = m_l_3.outer_faces() # Boundary list of Layer 3
bd_norm_3 = m_l_3.normal_of_faces(bd_3) # Normal vector for Boundary of Layer 3
bd_31_tf = abs(bd_norm_3[2,:]) < 1e-14
bd_32_tf = abs(bd_norm_3[2,:]-1) < 1e-14
bd_33_tf = abs(bd_norm_3[2,:]+1) < 1e-14
bd_31 = np.compress(bd_31_tf, bd_3, axis=1)
bd_32 = np.compress(bd_32_tf, bd_3, axis=1)
bd_33 = np.compress(bd_33_tf, bd_3, axis=1)
Boundary_Region_31 = 31; Boundary_Region_32 = 32; Boundary_Region_33 = 33;
m_l_3.set_region(Boundary_Region_31, bd_31)
m_l_3.set_region(Boundary_Region_32, bd_32)
m_l_3.set_region(Boundary_Region_33, bd_33)

##### Assembly procedure

# Stiffness matrix
# K_11 = gf.asm_linear_elasticity(mim_1, mfu_1, mfd_1, np.repeat([Lambda[0]], mfd_1.nbdof()), np.repeat([Mu[0]], mfd_1.nbdof()));
# K_2 = gf.asm_linear_elasticity(mim_2, mfu_2, mfd_2, np.repeat([Lambda[1]], mfd_2.nbdof()), np.repeat([Mu[1]], mfd_2.nbdof()));
# K_3 = gf.asm_linear_elasticity(mim_3, mfu_3, mfd_3, np.repeat([Lambda[2]], mfd_3.nbdof()), np.repeat([Mu[2]], mfd_3.nbdof()));
md_1 = gf.Model("real"); md_1.add_fem_variable("u_1", mfu_1); 
md_1.add_initialized_data('lambda', Lambda[0]); md_1.add_initialized_data('mu', Mu[0])
K_1 = gf.asm_generic(mim_1, 2, "lambda*Div_u_1*Div_Test_u_1 + mu*(Grad_u_1 + Grad_u_1'):Grad_Test_u_1", -1, md_1)

md_2 = gf.Model("real"); md_2.add_fem_variable("u_2", mfu_2)
md_2.add_initialized_data('lambda', Lambda[1]); md_2.add_initialized_data('mu', Mu[1])
K_2 = gf.asm_generic(mim_2, 2, "lambda*Div_u_2*Div_Test_u_2 + mu*(Grad_u_2 + Grad_u_2'):Grad_Test_u_2", -1, md_2)

md_3 = gf.Model("real"); md_3.add_fem_variable("u_3", mfu_3) 
md_3.add_initialized_data('lambda', Lambda[2]); md_3.add_initialized_data('mu', Mu[2])
K_3 = gf.asm_generic(mim_3, 2, "lambda*Div_u_3*Div_Test_u_3 + mu*(Grad_u_3 + Grad_u_3'):Grad_Test_u_3", -1, md_3)

# Dirichlet conditions
dof_list_11 = mfu_1.dof_on_region(Boundary_Region_11) # Finite element nodes on boundary region 11
local_list_11 = mfu_1.basic_dof_nodes(dof_list_11) # Coordinates of finite element nodes on boundary region 11
for i in dof_list_11: K_1[i, i] = Dirichlet_Penalization_coefficient

dof_list_21 = mfu_2.dof_on_region(Boundary_Region_21) # Finite element nodes on boundary region 21
local_list_21 = mfu_2.basic_dof_nodes(dof_list_21) # Coordinates of finite element nodes on boundary region 21
for i in dof_list_21: K_2[i, i] = Dirichlet_Penalization_coefficient

dof_list_31 = mfu_3.dof_on_region([Boundary_Region_31,Boundary_Region_33]) # Finite element nodes on boundary region 31
local_list_31 = mfu_3.basic_dof_nodes(dof_list_31) # Coordinates of finite element nodes on boundary region 31
for i in dof_list_31: K_3[i, i] = Dirichlet_Penalization_coefficient

size_K1 = K_1.size()[0]; size_K2 = K_2.size()[0]; size_K3 = K_3.size()[0]; 
size_K = size_K1 + size_K2 + size_K3
K = gf.Spmat('identity', size_K)
K.assign(range(size_K1), range(size_K1), K_1) 
K.assign(range(size_K1, size_K1+size_K2), range(size_K1, size_K1+size_K2), K_2)
K.assign(range(size_K1+size_K2, size_K1+size_K2+size_K3), range(size_K1+size_K2, size_K1+size_K2+size_K3), K_3)

## Load Vector
mdf_1 = gf.Model("real"); mdf_1.add_fem_variable("u_1", mfu_1);
mdf_1.add_macro('pressure', pressure)
mdf_1.add_macro('gravity', gravity_1)
f_2 = gf.asm_generic(mim_1, 1, "pressure'*u_1", Boundary_Region_12, mdf_1)
f0_1 = gf.asm_generic(mim_1, 1, "gravity'*u_1", -1, mdf_1)

mdf_2 = gf.Model("real"); mdf_2.add_fem_variable("u_2", mfu_2);
mdf_2.add_macro('gravity', gravity_2)
f0_2 = gf.asm_generic(mim_2, 1, "gravity'*u_2", -1, mdf_2)

mdf_3 = gf.Model("real"); mdf_3.add_fem_variable("u_3", mfu_3);
mdf_3.add_macro('gravity', gravity_3)
f0_3 = gf.asm_generic(mim_3, 1, "gravity'*u_3", -1, mdf_3)

f_x = np.append(f0_1+f_2,f0_2)
f = np.append(f_x,f0_3)

## Matrix G
dof_list_13 = mfu_1.dof_on_region(Boundary_Region_13)
local_list_13 = mfu_1.basic_dof_nodes(dof_list_13)

dof_list_22 = mfu_2.dof_on_region(Boundary_Region_22)
local_list_22 = mfu_2.basic_dof_nodes(dof_list_22)

dof_list_23 = mfu_2.dof_on_region(Boundary_Region_23)
local_list_23 = mfu_2.basic_dof_nodes(dof_list_23)

dof_list_32 = mfu_3.dof_on_region(Boundary_Region_32)
local_list_32 = mfu_3.basic_dof_nodes(dof_list_32)

# GN_1
mdn_13 = gf.Model("real"); 
mdn_13.add_fem_variable("u_1", mfu_1); 
mdn_13.add_fem_variable("mu_N1", mfc_N1); 
Gn_13_0 = gf.asm_generic(mim_1, 2, "(mu_N1)*(u_1'*Normal)", Boundary_Region_13, mdn_13)
# notice that the vector is [mu_N1, u_1]
dof_mdn_N13 = mfc_N1.dof_on_region(Boundary_Region_13)
local_mdn_N13 = mfc_N1.basic_dof_nodes(dof_mdn_N13)
Gn_13_1 = gf.Spmat('copy', Gn_13_0 , range(0,mfc_N1.nbdof()), range(mfc_N1.nbdof(),mfc_N1.nbdof()+mfu_1.nbdof() ))
Gn_13_2 = gf.Spmat('copy', Gn_13_1 , dof_mdn_N13, range(0,mfu_1.nbdof()))

mdn_22 = gf.Model("real"); 
mdn_22.add_fem_variable("u_2", mfu_2); 
mdn_22.add_fem_variable("mu_N2", mfc_N2); 
Gn_22_0 = gf.asm_generic(mim_2, 2, "(mu_N2)*(u_2'*Normal)", Boundary_Region_22, mdn_22)
dof_mdn_N22 = mfc_N2.dof_on_region(Boundary_Region_22)
local_mdn_N22 = mfc_N2.basic_dof_nodes(dof_mdn_N22)
Gn_22_1 = gf.Spmat('copy', Gn_22_0 , range(0,mfc_N2.nbdof()), range(mfc_N2.nbdof(),mfc_N2.nbdof()+mfu_2.nbdof() ))
Gn_22_2 = gf.Spmat('copy', Gn_22_1 , dof_mdn_N22, range(0,mfu_2.nbdof()))

Gn_13_2_size = Gn_13_2.size(); Gn_22_2_size = Gn_22_2.size();
if Gn_13_2_size[0] == Gn_22_2_size[0]:
    GN_1 = gf.Spmat('empty', Gn_13_2_size[0], size_K)
    GN_1.add(range(0,Gn_13_2_size[0]), range(0,size_K1), Gn_13_2)
    GN_1.add(range(0,Gn_22_2_size[0]), range(size_K1, size_K1+size_K2), Gn_22_2)
else:
    print('Error')
## Note that the direction of integration is always positive during computer integration, so there is no need to add a negative sign when assembling.

# GN_2
mdn_23 = gf.Model("real"); 
mdn_23.add_fem_variable("u_2", mfu_2); 
mdn_23.add_fem_variable("mu_N2", mfc_N2); 
Gn_23_0 = gf.asm_generic(mim_2, 2, "(mu_N2)*(u_2'*Normal)", Boundary_Region_23, mdn_23)
dof_mdn_N23 = mfc_N2.dof_on_region(Boundary_Region_23)
local_mdn_N23 = mfc_N2.basic_dof_nodes(dof_mdn_N23)
Gn_23_1 = gf.Spmat('copy', Gn_23_0 , range(0,mfc_N2.nbdof()), range(mfc_N2.nbdof(),mfc_N2.nbdof()+mfu_2.nbdof() ))
Gn_23_2 = gf.Spmat('copy', Gn_23_1 , dof_mdn_N23, range(0,mfu_2.nbdof()))

mdn_32 = gf.Model("real"); 
mdn_32.add_fem_variable("u_3", mfu_3); 
mdn_32.add_fem_variable("mu_N3", mfc_N3); 
Gn_32_0 = gf.asm_generic(mim_3, 2, "(mu_N3)*(u_3'*Normal)", Boundary_Region_32, mdn_32)
dof_mdn_N32 = mfc_N3.dof_on_region(Boundary_Region_32)
local_mdn_N32 = mfc_N3.basic_dof_nodes(dof_mdn_N32)
Gn_32_1 = gf.Spmat('copy', Gn_32_0 , range(0,mfc_N3.nbdof()), range(mfc_N3.nbdof(),mfc_N3.nbdof()+mfu_3.nbdof() ))
Gn_32_2 = gf.Spmat('copy', Gn_32_1 , dof_mdn_N32, range(0,mfu_3.nbdof()))

Gn_23_2_size = Gn_23_2.size(); Gn_32_2_size = Gn_32_2.size();
if Gn_23_2_size[0] == Gn_32_2_size[0]:
    GN_2 = gf.Spmat('empty', Gn_23_2_size[0], size_K)
    GN_2.add(range(0,Gn_23_2_size[0]), range(size_K1,size_K1+size_K2), Gn_23_2)
    GN_2.add(range(0,Gn_32_2_size[0]), range(size_K1+size_K2, size_K), Gn_32_2)
else:
    print('Error')

# GT_1
mdt_13 = gf.Model("real"); 
mdt_13.add_fem_variable("u_1", mfu_1); 
mdt_13.add_fem_variable("mu_T1", mfc_T1); 
Gt_13_0 = gf.asm_generic(mim_1, 2, "mu_T1'*(u_1-(u_1'*Normal)*Normal)", Boundary_Region_13, mdt_13)
# notice that the vector is [mu_N1, u_1]
dof_mdt_T13 = mfc_T1.dof_on_region(Boundary_Region_13)
local_mdt_T13 = mfc_T1.basic_dof_nodes(dof_mdt_T13)
Gt_13_1 = gf.Spmat('copy', Gt_13_0 , range(0,mfc_T1.nbdof()), range(mfc_T1.nbdof(),mfc_T1.nbdof()+mfu_1.nbdof() ))
Gt_13_2 = gf.Spmat('copy', Gt_13_1 , dof_mdt_T13, range(0,mfu_1.nbdof()))

mdt_22 = gf.Model("real"); 
mdt_22.add_fem_variable("u_2", mfu_2); 
mdt_22.add_fem_variable("mu_T2", mfc_T2); 
Gt_22_0 = gf.asm_generic(mim_2, 2, "mu_T2'*(u_2-(u_2'*Normal)*Normal)", Boundary_Region_22, mdt_22)
dof_mdt_T22 = mfc_T2.dof_on_region(Boundary_Region_22)
local_mdt_T22 = mfc_T2.basic_dof_nodes(dof_mdt_T22)
Gt_22_1 = gf.Spmat('copy', Gt_22_0 , range(0,mfc_T2.nbdof()), range(mfc_T2.nbdof(),mfc_T2.nbdof()+mfu_2.nbdof() ))
Gt_22_2 = gf.Spmat('copy', Gt_22_1 , dof_mdt_T22, range(0,mfu_2.nbdof()))

Gt_13_2_size = Gt_13_2.size(); Gt_22_2_size = Gt_22_2.size();
if Gt_13_2_size[0] == Gt_22_2_size[0]:
    GT_1 = gf.Spmat('empty', Gt_13_2_size[0], size_K)
    GT_1.add(range(0,Gt_13_2_size[0]), range(0,size_K1), Gt_13_2)
    GT_1.add(range(0,Gt_22_2_size[0]), range(size_K1, size_K1+size_K2), Gt_22_2)
else:
    print('Error')

# GT_2
mdt_23 = gf.Model("real"); 
mdt_23.add_fem_variable("u_2", mfu_2); 
mdt_23.add_fem_variable("mu_T2", mfc_T2); 
Gt_23_0 = gf.asm_generic(mim_2, 2, "mu_T2'*(u_2-(u_2'*Normal)*Normal)", Boundary_Region_23, mdt_23)
# notice that the vector is [mu_N1, u_1]
dof_mdt_T23 = mfc_T2.dof_on_region(Boundary_Region_23)
local_mdt_T23 = mfc_T2.basic_dof_nodes(dof_mdt_T23)
Gt_23_1 = gf.Spmat('copy', Gt_23_0 , range(0,mfc_T2.nbdof()), range(mfc_T2.nbdof(),mfc_T2.nbdof()+mfu_2.nbdof() ))
Gt_23_2 = gf.Spmat('copy', Gt_23_1 , dof_mdt_T23, range(0,mfu_2.nbdof()))

mdt_32 = gf.Model("real"); 
mdt_32.add_fem_variable("u_3", mfu_3); 
mdt_32.add_fem_variable("mu_T3", mfc_T3); 
Gt_32_0 = gf.asm_generic(mim_3, 2, "mu_T3'*(u_3-(u_3'*Normal)*Normal)", Boundary_Region_32, mdt_32)
dof_mdt_T32 = mfc_T3.dof_on_region(Boundary_Region_32)
local_mdt_T32 = mfc_T3.basic_dof_nodes(dof_mdt_T32)
Gt_32_1 = gf.Spmat('copy', Gt_32_0 , range(0,mfc_T3.nbdof()), range(mfc_T3.nbdof(), mfc_T3.nbdof()+mfu_3.nbdof() ))
Gt_32_2 = gf.Spmat('copy', Gt_32_1 , dof_mdt_T32, range(0,mfu_3.nbdof()))

Gt_23_2_size = Gt_23_2.size(); Gt_32_2_size = Gt_32_2.size();
if Gt_23_2_size[0] == Gt_32_2_size[0]:
    GT_2 = gf.Spmat('empty', Gt_23_2_size[0], size_K)
    GT_2.add(range(0,Gt_23_2_size[0]), range(size_K1, size_K1+size_K2), Gt_23_2)
    GT_2.add(range(0,Gt_32_2_size[0]), range(size_K1+size_K2, size_K), Gt_32_2)
else:
    print('Error')

# G
size_GN_1 = GN_1.size(); size_GN_2 = GN_2.size(); 
size_GT_1 = GT_1.size(); size_GT_2 = GT_2.size(); 
size_G = size_GN_1[0] + size_GN_2[0] + size_GT_1[0] + size_GT_2[0]
G = gf.Spmat('empty', size_G, size_K)
G.add(range(0,size_GN_1[0]), range(size_K), GN_1)
G.add(range(size_GN_1[0], size_GN_1[0]+size_GN_2[0]), range(size_K), GN_2)
G.add(range(size_GN_1[0]+size_GN_2[0], size_GN_1[0]+size_GN_2[0]+size_GT_1[0]), range(size_K), GT_1)
G.add(range(size_GN_1[0]+size_GN_2[0]+size_GT_1[0], size_G), range(size_K), GT_2)


##### Calculation

K.to_csc(); G.to_csc();
K.save('Matrix-Market', 'K.txt'); G.save('Matrix-Market', 'G.txt')
with open("K.txt", "r") as k:
    K_txt = k.read()
K_coo = mmread(StringIO(K_txt))
K_csc = K_coo.tocsc()
with open("G.txt", "r") as g:
    G_txt = g.read()
G_coo = mmread(StringIO(G_txt)); GT_coo = np.transpose(G_coo)
G_csc = G_coo.tocsc(); GT_csc = GT_coo.tocsc()

# K_inv = inv(K_csc)

# M = G*K^{-1}*G^T
K_inv_G_T = spsolve(K_csc, GT_csc)
# K_inv_G_T = K_inv @ GT_csc
M = G_csc @ K_inv_G_T

# b = G*K^{-1}*f
K_inv_f = spsolve(K_csc, f)
# K_inv_f = K_inv @ f
b = G_csc @ K_inv_f

# Optimization: mu = argmin 0.5*mu^T*M*mu - mu^T*b
M_dense = M.todense()
size_GN = size_GN_1[0] + size_GN_2[0]
'''
fun = lambda mu: 0.5* np.transpose(mu) @ M_dense @ mu - np.transpose(mu) @ b
mu0 = np.repeat(1e-5,size_G)
for i in range(0,int(size_GT_1[0]/3+size_GT_2[0]/3)):
    mu0[size_GN+i*3+2] = 0
bnds = ()
for i in range(0,size_GN):
    bnds = bnds + ((0, 1e20),)
    
for i in range(0,int(size_GT_1[0]/3)):
    bound_GT_1_x = fun_g_1(local_mdt_T13[:,i])
    bound_GT_1_y = fun_g_1(local_mdt_T13[:,i+1])
    bnds = bnds + ((-bound_GT_1_x, bound_GT_1_x),)
    bnds = bnds + ((-bound_GT_1_y, bound_GT_1_y),)
    bnds = bnds + ((-1e-24, 1e-24),)

for i in range(0,int(size_GT_2[0]/3)):
    bound_GT_2_x = fun_g_2(local_mdt_T23[:,i])
    bound_GT_2_y = fun_g_2(local_mdt_T23[:,i+1])
    bnds = bnds + ((-bound_GT_2_x, bound_GT_2_x),)
    bnds = bnds + ((-bound_GT_2_y, bound_GT_2_y),)
    bnds = bnds + ((-1e-24, 1e-24),)
'''
# res = minimize(fun, mu0, method='Nelder-Mead', bounds=bnds, tol=1e-6)

# Constrain Condition (need to be revised: vector: 1-norm)

E_11 = -np.eye(size_GN_1[0]+size_GN_2[0])
E_12 = np.zeros((size_GN_1[0]+size_GN_2[0], size_GT_1[0]+size_GT_2[0]))
E_21 = np.zeros((size_GN_1[0]+size_GN_2[0], int((size_GT_1[0]+size_GT_2[0])/3)))
E_22 = np.zeros((int((size_GT_1[0]+size_GT_2[0])/3), size_GT_1[0]+size_GT_2[0]))
E_32 = np.zeros((int((size_GT_1[0]+size_GT_2[0])/3), size_GT_1[0]+size_GT_2[0]))
E_42 = np.zeros((int((size_GT_1[0]+size_GT_2[0])/3), size_GT_1[0]+size_GT_2[0]))
E_52 = np.zeros((int((size_GT_1[0]+size_GT_2[0])/3), size_GT_1[0]+size_GT_2[0]))
E_62 = np.zeros((int((size_GT_1[0]+size_GT_2[0])/3), size_GT_1[0]+size_GT_2[0]))
E_72 = np.zeros((int((size_GT_1[0]+size_GT_2[0])/3), size_GT_1[0]+size_GT_2[0]))
E_82 = np.zeros((int((size_GT_1[0]+size_GT_2[0])/3), size_GT_1[0]+size_GT_2[0]))
E_92 = np.zeros((int((size_GT_1[0]+size_GT_2[0])/3), size_GT_1[0]+size_GT_2[0]))
for i in range(0,int((size_GT_1[0]+size_GT_2[0])/3)):
    E_22[i,[3*i,3*i+1,3*i+2]] = [1.0,1.0,1.0];
    E_32[i,[3*i,3*i+1,3*i+2]] = [-1.0,1.0,1.0];
    E_42[i,[3*i,3*i+1,3*i+2]] = [1.0,-1.0,1.0];
    E_52[i,[3*i,3*i+1,3*i+2]] = [1.0,1.0,-1.0];
    E_62[i,[3*i,3*i+1,3*i+2]] = [1.0,-1.0,-1.0];
    E_72[i,[3*i,3*i+1,3*i+2]] = [-1.0,1.0,-1.0];
    E_82[i,[3*i,3*i+1,3*i+2]] = [-1.0,-1.0,1.0];
    E_92[i,[3*i,3*i+1,3*i+2]] = [-1.0,-1.0,-1.0];
    
E_1 = np.concatenate((E_11,E_12),axis=1)
E_2 = np.concatenate((E_21.T,E_22),axis=1)
E_3 = np.concatenate((E_21.T,E_32),axis=1)
E_4 = np.concatenate((E_21.T,E_42),axis=1)
E_5 = np.concatenate((E_21.T,E_52),axis=1)
E_6 = np.concatenate((E_21.T,E_62),axis=1)
E_7 = np.concatenate((E_21.T,E_72),axis=1)
E_8 = np.concatenate((E_21.T,E_82),axis=1)
E_9 = np.concatenate((E_21.T,E_92),axis=1)
E = np.concatenate((E_1,E_2,E_3,E_4,E_5,E_6,E_7,E_8,E_9),axis=0)

h_N = np.repeat(0.0,size_GN_1[0]+size_GN_2[0])
h_T = np.repeat(0.0,int((size_GT_1[0]+size_GT_2[0])/3))
for i in range(0,int(size_GT_1[0]/3)):
    bound_GT_1 = fun_g_1(local_mdt_T13[:,3*i])
    h_T[i] = bound_GT_1;
for i in range(0,int(size_GT_2[0]/3)):
    bound_GT_2 = fun_g_2(local_mdt_T23[:,3*i])
    h_T[int(size_GT_1[0]/3)+i] = bound_GT_2;
h = np.concatenate((h_N,h_T,h_T,h_T,h_T,h_T,h_T,h_T,h_T),axis=0);

M_matrix = matrix(M_dense)
b_matrix = matrix(b)
E_matrix = matrix(E)
h_matrix = matrix(h)

result = qp(M_matrix, -b_matrix, E_matrix, h_matrix)
lambdas = result['x']
lambdan = np.array(lambdas)
lambdaa = lambdan.ravel()
fl = f - (GT_csc @ lambdaa)
u = spsolve(K_csc, fl )
# u = K_inv @ fl 

# Displacement 
u_1 = u[np.arange(0,size_K1)]
u_2 = u[np.arange(size_K1,size_K1+size_K2)]
u_3 = u[np.arange(size_K1+size_K2,size_K)]

# Displacement of contact zone
Trans_local_1322 = local_list_13 - local_list_22
T_nozeor_1322 = np.nonzero(Trans_local_1322[0:1,:])
T_size_1322 = np.size(T_nozeor_1322[0])
T_1322_p =[]; T_1322_n =[];
for i in np.arange(0,T_size_1322):
    if Trans_local_1322[T_nozeor_1322[0][i],T_nozeor_1322[1][i]] > 0:
        T_1322_p.append(T_nozeor_1322[1][i])
    if Trans_local_1322[T_nozeor_1322[0][i],T_nozeor_1322[1][i]] < 0:
        T_1322_n.append(T_nozeor_1322[1][i])
u_13 = u_1[dof_list_13]
u_22 = u_2[dof_list_22]
for i in np.arange(0,len(T_1322_p)):
    u_T = u_22[T_1322_p[i]]
    u_22[T_1322_p[i]] = u_22[T_1322_n[i]]
    u_22[T_1322_n[i]] = u_T
u_13_u_22 = u_13 - u_22
u_1322 = np.repeat(0.0,size_K1)
for i in np.arange(len(u_13_u_22)):
    u_1322[dof_list_13[i]] = u_13_u_22[i]

Trans_local_2332 = local_list_23 - local_list_32
T_nozeor_2332 = np.nonzero(Trans_local_2332[0:1,:])
T_size_2332 = np.size(T_nozeor_2332[0])
T_2332_p =[]; T_2332_n =[];
for i in np.arange(0,T_size_2332):
    if Trans_local_2332[T_nozeor_2332[0][i],T_nozeor_2332[1][i]] > 0:
        T_2332_p.append(T_nozeor_2332[1][i])
    if Trans_local_2332[T_nozeor_2332[0][i],T_nozeor_2332[1][i]] < 0:
        T_2332_n.append(T_nozeor_2332[1][i])
u_23 = u_2[dof_list_23]
u_32 = u_3[dof_list_32]
for i in np.arange(0,len(T_2332_p)):
    u_T = u_32[T_2332_p[i]]
    u_32[T_2332_p[i]] = u_32[T_2332_n[i]]
    u_32[T_2332_n[i]] = u_T
u_23_u_32 = u_23 - u_32
u_2332 = np.repeat(0.0,size_K2)
for i in np.arange(len(u_23_u_32)):
    u_2332[dof_list_23[i]] = u_23_u_32[i]

# Save mesh
u_1_Matrix = np.reshape(u_1, (-1,1)); mmwrite("u_1", u_1_Matrix);
u_2_Matrix = np.reshape(u_2, (-1,1)); mmwrite("u_2", u_2_Matrix);
u_3_Matrix = np.reshape(u_3, (-1,1)); mmwrite("u_3", u_3_Matrix);

s1 = gf.Slice(('boundary',), mfu_1, 1)
s1.export_to_vtk('Dis_1.vtk', 'ascii', mfu_1, u_1, 'Displacement')
s2 = gf.Slice(('boundary',), mfu_2, 1)
s2.export_to_vtk('Dis_2.vtk', 'ascii', mfu_2, u_2, 'Displacement')
s3 = gf.Slice(('boundary',), mfu_3, 1)
s3.export_to_vtk('Dis_3.vtk', 'ascii', mfu_3, u_3, 'Displacement')

s13 = gf.Slice(('boundary',), mfu_1, 1)
s13.export_to_vtk('Dis_13.vtk', 'ascii', mfu_1, u_1322, 'Displacement')
s23 = gf.Slice(('boundary',), mfu_2, 1)
s23.export_to_vtk('Dis_23.vtk', 'ascii', mfu_2, u_2332, 'Displacement')

