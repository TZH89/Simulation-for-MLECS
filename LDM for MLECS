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
from scipy.io import hb_write, hb_read
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
# from scipy.optimize import minimize
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
fun_g_1 = '0.2' #lambda x: 0.2
fun_g_2 = '0.05' #lambda x: 0.05

degree = 1 # Order k of finite element polynomial space
degree_lambda = 1 # Definition of boundary element space

Dirichlet_Penalization_coefficient = 1e30

Thetas = [0.04] # [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.082, 0.084, 0.086, 0.088, 0.0885, 0.089]
tolerance = 1e-4
toleranceNum = 1e5

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

## Stiffness matrix
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

K_1.save('Matrix-Market', 'K_1.txt'); 
with open("K_1.txt", "r") as k:
    K1_txt = k.read()
K1 = mmread(StringIO(K1_txt))
K1 = K1.tocsc(); del K1_txt;

K_2.save('Matrix-Market', 'K_2.txt'); 
with open("K_2.txt", "r") as k:
    K2_txt = k.read()
K2 = mmread(StringIO(K2_txt))
K2 = K2.tocsc(); del K2_txt;

K_3.save('Matrix-Market', 'K_3.txt'); 
with open("K_3.txt", "r") as k:
    K3_txt = k.read()
K3 = mmread(StringIO(K3_txt))
K3 = K3.tocsc(); del K3_txt;
del k;

#K1_inv = inv(K1); K2_inv = inv(K2); K3_inv = inv(K3);
#K1_inv = spsolve(K1,np.eye(size_K1)); 
#K2_inv = spsolve(K2,np.eye(size_K2));
#K3_inv = spsolve(K3,np.eye(size_K3));

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

## Matrix Ci, Bi
dof_list_13 = mfu_1.dof_on_region(Boundary_Region_13)
local_list_13 = mfu_1.basic_dof_nodes(dof_list_13)

dof_list_22 = mfu_2.dof_on_region(Boundary_Region_22)
local_list_22 = mfu_2.basic_dof_nodes(dof_list_22)

dof_list_23 = mfu_2.dof_on_region(Boundary_Region_23)
local_list_23 = mfu_2.basic_dof_nodes(dof_list_23)

dof_list_32 = mfu_3.dof_on_region(Boundary_Region_32)
local_list_32 = mfu_3.basic_dof_nodes(dof_list_32)

# N_1, I11, I13, T1, B1
Gn_13 = gf.asm_generic(mim_1, 1, "(u_1'*Normal)", Boundary_Region_13, md_1)
num_dof_13 = int(len(dof_list_13)/3) # number of FE nodes on boundary 13
N13 = np.zeros([num_dof_13 , size_K1]); 
I11 = np.zeros([size_K1 , size_K1])
I13 = np.zeros([num_dof_13*3 , size_K1])
for i in range(num_dof_13):
    k13 = dof_list_13[[3*i,3*i+1,3*i+2]]
    n13 = Gn_13[k13]
    n13 = n13/np.linalg.norm(n13)
    for j in range(3):
        N13[i, k13[j]] = n13[j]
        I11[k13[j], k13[j]] = 1.0
        I13[3*i+j, k13[j]] = 1.0

T1 = I11 - N13.T @ N13
I13_T1 = I13@T1
B1 = np.concatenate((N13,I13_T1),axis=0)
    
# N_2, I21, I22, I23, I24, B2
Gn_23 = gf.asm_generic(mim_2, 1, "(u_2'*Normal)", Boundary_Region_23, md_2)
num_dof_23 = int(len(dof_list_23)/3) # number of FE nodes on boundary 23
N23 = np.zeros([num_dof_23 , size_K2])
I21 = np.zeros([size_K2, size_K2])
I23 = np.zeros([num_dof_23*3 , size_K2])
for i in range(num_dof_23):
    k23 = dof_list_23[[3*i,3*i+1,3*i+2]]
    n23 = Gn_23[k23]
    n23 = n23/np.linalg.norm(n23)
    for j in range(3):
        N23[i, k23[j]] = n23[j]
        I21[k23[j], k23[j]]= 1.0
        I23[3*i+j, k23[j]] = 1.0

num_dof_22 = int(len(dof_list_22)/3) # number of FE nodes on boundary 23
I22 = np.zeros([size_K2, size_K2])
I24 = np.zeros([num_dof_22*3 , size_K2])
for i in range(num_dof_22):
    k22 = dof_list_22[[3*i,3*i+1,3*i+2]]
    for j in range(3):
        I22[k22[j], k22[j]] = 1.0
        I24[3*i+j, k22[j]] = 1.0
        
T2 = I21 - N23.T@ N23
I23_T2 = I23@ T2
B2 = np.concatenate((N23,I23_T2,I24),axis=0)

# I32, I34, B3
num_dof_32 = int(len(dof_list_32)/3) # number of FE nodes on boundary 23
I32 = np.zeros([size_K3, size_K3])
I34 = np.zeros([num_dof_32*3 , size_K3])
for i in range(num_dof_32):
    k32 = dof_list_32[[3*i,3*i+1,3*i+2]]
    for j in range(3):
        I32[k32[j], k32[j]] = 1.0
        I34[3*i+j, k32[j]] = 1.0
B3 = I34

## Matrix C1, C2, C3 and Vector Ab1, Ab2, Ab3
#C1 = B1 @ K1_inv @ B1.T
C1 = B1 @ spsolve(K1,B1.T) 
#C2 = B2 @ K2_inv @ B2.T
C2 = B2 @ spsolve(K2,B2.T)
#C3 = B3 @ K3_inv @ B3.T
C3 = B3 @ spsolve(K3,B3.T)
#Ab1 = K1_inv @ (f0_1+f_2)
Ab1 = spsolve(K1, (f0_1+f_2))
#Ab2 = K2_inv @ f0_2
Ab2 = spsolve(K2, f0_2)
#Ab3 = K3_inv @ f0_3
Ab3 = spsolve(K3, f0_3)

## vector g13, g23
md_1.add_macro('g13', fun_g_1)
g_13 = gf.asm_generic(mim_1, 1, "(u_1(1)+u_1(2)+u_1(3))*g13", Boundary_Region_13, md_1)
g_13 = I13@g_13
md_2.add_macro('g23', fun_g_2)
g_23 = gf.asm_generic(mim_2, 1, "(u_2(1)+u_2(2)+u_2(3))*g23", Boundary_Region_23, md_2)
g_23 = I23@g_23

#### Algorithm LDM

## constrain conditions
E1_11 = np.eye(np.size(N13,0))
E1_12 = np.zeros([np.size(N13,0),np.size(I13_T1,0)])
E1_22 = np.eye(np.size(I13_T1,0))
E1_1 = np.concatenate((-E1_11, E1_12),axis=1)
E1_2 = np.concatenate((E1_12.T, E1_22),axis=1)
E1_3 = np.concatenate((E1_12.T, -E1_22),axis=1)
E1 = np.concatenate((E1_1, E1_2, E1_3),axis=0)
h1_1 = np.repeat(0.0,np.size(N13,0))
h1 = np.concatenate((h1_1, g_13 , g_13),axis=0)

E2_11 = np.eye(np.size(N23,0))
E2_12 = np.zeros([np.size(N23,0),np.size(I23_T2,0)])
E2_13 = np.zeros([np.size(N23,0),np.size(I24,0)])
E2_22 = np.eye(np.size(I23_T2,0))
E2_23 = np.zeros([np.size(I23_T2,0),np.size(I24,0)])
E2_1 = np.concatenate((-E2_11, E2_12, E2_13),axis=1)
E2_2 = np.concatenate((E2_12.T, E2_22, E2_23),axis=1)
E2_3 = np.concatenate((E2_12.T, -E2_22, E2_23),axis=1)
E2 = np.concatenate((E2_1, E2_2, E2_3),axis=0)
h2_1 = np.repeat(0.0,np.size(N23,0))
h2 = np.concatenate((h2_1, g_23 , g_23),axis=0)

## optimization
num_iterations=[]; 

for theta in Thetas:
    ## initial value
    vlambdak1 = np.repeat(1e-21, size_K1)
    plambdak1 = np.repeat(1e-21, size_K1)
    vlambdak2 = np.repeat(1e-21, size_K2)
    plambdak2 = np.repeat(1e-21, size_K2)
    error = 1; num_it=0;
    while error > tolerance:
        lambdab1 = vlambdak1 - Ab1
        N13lambdab1 = N13 @ lambdab1
        I13T11lambdab1 = I13_T1 @ lambdab1
        d1 = np.concatenate((N13lambdab1, I13T11lambdab1),axis=0)
        
        lambdab2 = vlambdak2 - Ab2;
        N23lambdab2 = N23 @ lambdab2;
        I23T2lambdab2 = I23_T2 @ lambdab2;
        lambda1I13 = I13 @ vlambdak1;
        Ab2I24 = I24 @ Ab2;
        lambdab1Ab2 = lambda1I13 - Ab2I24;
        d2 = np.concatenate((N23lambdab2, I23T2lambdab2, lambdab1Ab2),axis=0)
        
        lambda2I23 = I23 @ vlambdak2;
        Ab3I34 = I34 @ Ab3;
        d3 = lambda2I23 - Ab3I34;
        
        omega_1 = qp(matrix(C1), matrix(d1), matrix(E1), matrix(h1), options = {'reltol' : 1e-9})
        omega_1 = omega_1['x']; omega_1 = np.array(omega_1); omega_1 = omega_1.ravel();
        omega_2 = qp(matrix(C2), matrix(d2), matrix(E2), matrix(h2), options = {'reltol' : 1e-9})
        omega_2 = omega_2['x']; omega_2 = np.array(omega_2); omega_2 = omega_2.ravel();
        omega_3 = qp(matrix(C3), matrix(d3), options = {'reltol' : 1e-9})
        omega_3 = omega_3['x']; omega_3 = np.array(omega_3); omega_3 = omega_3.ravel();
        
        pq1 = I24 @ (B2.T) @ omega_2 + I13 @ B1.T @ omega_1;
        pq2 = I34 @ B3.T @ omega_3 + I23 @ B2.T @ omega_2;
        
        #p2 = K2_inv @ (-0.5*(I24.T) @ pq1);
        p2 = spsolve(K2, (-0.5*(I24.T) @ pq1));
        #q1 = K1_inv @ (-0.5*(I13.T) @ pq1);
        q1 = spsolve(K1, (-0.5*(I13.T) @ pq1));
        #p3 = K3_inv @ (-0.5*(I34.T) @ pq2);
        p3 = spsolve(K3, (-0.5*(I34.T) @ pq2));
        #q2 = K2_inv @ (-0.5*(I23.T) @ pq2);
        q2 = spsolve(K2, (-0.5*(I23.T) @ pq2));
        
        plambdak1 = vlambdak1; plambdak2 = vlambdak2;
        vlambdak1 = I13.T @ (I13@plambdak1 - theta*(I24@p2 + I13@q1));
        vlambdak2 = I23.T @ (I23@plambdak2 - theta*(I34@p3 + I23@q2));
        
        error = (np.linalg.norm(vlambdak1-plambdak1,ord=2) + np.linalg.norm(vlambdak2-plambdak2,ord=2))/(np.linalg.norm(vlambdak1,ord=2) + np.linalg.norm(vlambdak2,ord=2))
        num_it = num_it+1;
        print("[Hint] Number of iteration: %d, with theta: %f" %(num_it,theta))
        print("[Hint] Error of lambdav in iteration: %f" %(error))
        if num_it > toleranceNum:
            print('[Hint] The must number of iteration has benn reached!')
            break
    
    num_iterations.append(num_it)

# Displacement 
#u_1 = K1_inv @ (f0_1 + f_2 - (B1.T @ omega_1))
u_1 = spsolve(K1, (f0_1 + f_2 - (B1.T @ omega_1)) )
#u_2 = K2_inv @ (f0_2 - (B2.T @ omega_2))
u_2 = spsolve(K2, (f0_2 - (B2.T @ omega_2)) )
#u_3 = K3_inv @ (f0_3 - (B3.T @ omega_3))
u_3 = spsolve(K3, (f0_3 - (B3.T @ omega_3)) )

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
