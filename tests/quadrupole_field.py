import numpy as np
import time
import itertools
import warnings
from scipy.special import jv as besselj
from scipy.special import erfc
from scipy.interpolate import interp1d
from scipy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.sparse.linalg import gmres,LinearOperator

class Electric_Field():#{{{
    def __init__(self,box,xi,errortol,eps_p=None,points = None,dip = None,dip_pos=None):# {{{
        self.box = box
        self.xi = xi
        self.errortol = errortol

        self.dip = dip
        self.dip_pos = dip_pos

        if eps_p is None:
            self.calc_inter_dipole = False
            self.has_new_eps_p = False
        else:
            self.calc_inter_dipole = True
            self.eps_p = eps_p
            self.has_new_eps_p = True

        self.has_new_dip_pos = True
        self.has_new_points = True

        self._precalculations()
        # }}}
    def set_dipoles(self,dip,quad):# {{{
        self.dip = dip
        self.quad = quad
        # }}}
    def set_dip_pos(self,dip_pos):# {{{
        self.dip_pos = dip_pos
        self.has_new_dip_pos = True
    # }}}
    def set_points(self,points):# {{{
        self.points = points
        self.has_new_points = True
    # }}}
    def set_eps_p(self,eps_p):# {{{
        self.has_new_eps_p = True
        self.calc_inter_dipole = True
        self.eps_p = eps_p
        # }}}
    def calculate(self):# {{{
        if self.dip_pos is None:
            raise Exception("Dipole positions must be set before calculating electric fields")
        if self.dip is None:
            raise Exception("Dipoles must be set before calculating electric fields")
        if self.has_new_dip_pos or self.has_new_points:
            self._real_space_precalcs()
        if self.has_new_points:
            self._contract_precalcs()
            self.has_new_points = False
        if self.has_new_dip_pos:
            self._spread_precalcs()
            self.has_new_dip_pos = False
        if self.has_new_eps_p:
            self._real_self_precalcs()
            self.has_new_eps_p = False
        return self._electric_field()
    # }}}
    def _electric_field(self):#{{{
        P = self.dip
        Q = self.quad
        E_grid = self.E_grid
        G_grid = self.G_grid
        E_point= self.E_point 
        G_point= self.G_point 
    
        E_grid[...] = 0
        G_grid[...] = 0

        E_grid,G_grid = self._spread(E_grid,G_grid,P,Q)
        fE_grid = fftshift(fftn(E_grid,axes=(0,1,2),overwrite_x=True),axes=(0,1,2))
        fG_grid = fftshift(fftn(G_grid,axes=(0,1,2),overwrite_x=True),axes=(0,1,2))

        fEs_grid,fGs_grid = self._scale(fE_grid,fG_grid)
        Es_grid = ifftn(ifftshift(fEs_grid,axes=(0,1,2)),axes=(0,1,2),overwrite_x=True)
        Gs_grid = ifftn(ifftshift(fGs_grid,axes=(0,1,2)),axes=(0,1,2),overwrite_x=True)
    
        E_point[...] = 0
        G_point[...] = 0
        E_point,G_point = self._contract(E_point,G_point,Es_grid,Gs_grid)
        if getattr(self, 'debug_print', False):
            print("Python Reciprocal E_point:")
            print(E_point)
            print("Python Reciprocal G_point:")
            print(G_point)
        E_point,G_point = self._real_space(E_point,G_point,P,Q)
        return E_point, G_point
        # }}}

    def _spread(self,P_grid,Q_grid,P,Q):# {{{
        spread_coef = self.spread_coef
        spread_idxs =  self.spread_idxs

        num_spread = spread_idxs.shape[0]
    
        Espread = spread_coef[:,:,None]*P[:,None,:]
        Espread = Espread.reshape(num_spread,-1)
        np.add.at(P_grid,tuple(spread_idxs.T),Espread)

        Espread = spread_coef[:,:,None]*Q[:,None,:]
        Espread = Espread.reshape(num_spread,-1)
        np.add.at(Q_grid,tuple(spread_idxs.T),Espread)
    
        return P_grid, Q_grid
        # }}}
    def _scale(self,fE,fG):# {{{
        khat = self.khat
        Qfactor = self.Qfactor
        Qfactor_dot = self.Qfactor_dot

        np.multiply(fE,khat,out=fE)
        np.multiply(fG,Qfactor_dot,out=fG)

        fE_dot = fE[...,0]+fE[...,1]+fE[...,2]
        fG_dot = fG[...,0]+fG[...,1]+fG[...,2]+fG[...,3]+fG[...,4]

        Edot = self.scale_EP_coef*fE_dot + self.scale_EQ_coef*fG_dot
        Gdot = self.scale_GP_coef*fE_dot + self.scale_GQ_coef*fG_dot

        #np.multiply(khat,sum[...,None],out=fE)
        np.multiply(khat[...,0],Edot,out=fE[...,0])
        np.multiply(khat[...,1],Edot,out=fE[...,1])
        np.multiply(khat[...,2],Edot,out=fE[...,2])

        np.multiply(Qfactor[...,0],Gdot,out=fG[...,0])
        np.multiply(Qfactor[...,1],Gdot,out=fG[...,1])
        np.multiply(Qfactor[...,2],Gdot,out=fG[...,2])
        np.multiply(Qfactor[...,3],Gdot,out=fG[...,3])
        np.multiply(Qfactor[...,4],Gdot,out=fG[...,4])

        return fE,fG
    # }}}
    def _contract(self,E_point,G_point,Es_grid,Gs_grid):# {{{
        particle_index = self.particle_index
        contract_coef = self.contract_coef
        contract_idxs =  self.contract_idxs
        np.add.at(E_point,particle_index,contract_coef[:,None]*Es_grid[*contract_idxs.T])
        np.add.at(G_point,particle_index,contract_coef[:,None]*Gs_grid[*contract_idxs.T])
        return E_point, G_point
    # }}}
    def _real_space(self,E_point,G_point,P,Q):# {{{
        #Real
        p1 = self.p1
        p2 = self.p2
        r = self.delta
        rr = self.rr
        __rr = self.__rr
        para = self.para
        perp = self.perp
        I = self.I
        toQR = self.toQR
        toQL = self.toQL

        Q1 = self.Q1
        Q2 = self.Q2
        Q3 = self.Q3

        G1 = self.G1
        G2 = self.G2
        G3 = self.G3
        G4 = self.G4

        if self.calc_inter_dipole:
            E_point += self.self_P_coef * P
            E_point += self.self_perp * P
            G_point += self.self_Q_coef * Q
            G_point += self.self_G2 * Q


        P = P[p2]
        Q = Q[p2]
        r_P = np.sum(P*r,axis = -1)[:,None] #P2 x 1
        rr_P = r*r_P #P2 x 3
        rrr_P = __rr.T*r_P #P2 x 5
        Ir_P = I[None,:]*r_P #P2 x 5
        
        Pr_rP = np.zeros((P.shape[0], 5), dtype=P.dtype)
        Pr_rP[:, 0] = 2 * (P[:, 0] * r[:, 0] - P[:, 2] * r[:, 2])
        Pr_rP[:, 1] = 2 * (P[:, 0] * r[:, 1] + P[:, 1] * r[:, 0])
        Pr_rP[:, 2] = 2 * (P[:, 0] * r[:, 2] + P[:, 2] * r[:, 0])
        Pr_rP[:, 3] = 2 * (P[:, 1] * r[:, 1] - P[:, 2] * r[:, 2])
        Pr_rP[:, 4] = 2 * (P[:, 1] * r[:, 2] + P[:, 2] * r[:, 1])

        Q_r = np.array([Q[:,0]*r[:,0]+Q[:,1]*r[:,1]+Q[:,2]*r[:,2],
                        Q[:,1]*r[:,0]+Q[:,3]*r[:,1]+Q[:,4]*r[:,2],
                        Q[:,2]*r[:,0]+Q[:,4]*r[:,1]-(Q[:,0]+Q[:,3])*r[:,2]]).T #P2 x 3
        Q__rr = np.sum(Q*__rr.T,axis=-1)[:,None] #P2 x 1
        Irr__Q = I[None,:]*Q__rr #P2 x 5
        
        Q_rr_rr_Q = np.zeros((Q_r.shape[0], 5), dtype=Q_r.dtype)
        Q_rr_rr_Q[:, 0] = 2 * (Q_r[:, 0] * r[:, 0] - Q_r[:, 2] * r[:, 2])
        Q_rr_rr_Q[:, 1] = 2 * (Q_r[:, 0] * r[:, 1] + Q_r[:, 1] * r[:, 0])
        Q_rr_rr_Q[:, 2] = 2 * (Q_r[:, 0] * r[:, 2] + Q_r[:, 2] * r[:, 0])
        Q_rr_rr_Q[:, 3] = 2 * (Q_r[:, 1] * r[:, 1] - Q_r[:, 2] * r[:, 2])
        Q_rr_rr_Q[:, 4] = 2 * (Q_r[:, 1] * r[:, 2] + Q_r[:, 2] * r[:, 1])

        rrrr__Q = __rr.T*Q__rr #P2 x 5

        np.add.at(E_point,p1,perp*(P - rr_P) + para*rr_P)
        np.add.at(E_point,p1,-0.5*(Q1*Q__rr*r + 2*Q2*Q_r))
        np.add.at(G_point,p1,Q1*rrr_P + Q2*Pr_rP + (Q2 + Q3)*Ir_P)
        np.add.at(G_point,p1,0.5*(G1*Irr__Q + G2*Q + G3*(Irr__Q+2*Q_rr_rr_Q) + G4*rrrr__Q))
        return E_point, G_point
    # }}}
    def _spread_precalcs(self):# {{{
        num_dipoles = self.dip_pos.shape[0]
        num_spread = num_dipoles*self.offset.shape[0]
        grid_idxs = np.round(self.dip_pos/self.grid_spacing).astype(int)
        particle_grid_dist = grid_idxs*self.grid_spacing - self.dip_pos
        grid_effect_idxs = (grid_idxs[:,None,:] + self.offset[None,:,:] - 1) % self.num_grid
        spread_idxs = grid_effect_idxs.reshape(num_spread,3)

        grid_effect_dist = (particle_grid_dist[:,None,:] + self.offsetxyz[None,:,:])
        grid_effect_div_eta = np.sum(grid_effect_dist**2/self.spectral_split,axis = -1)
        spread_coef = (2*self.xi**2/np.pi)**(3/2)*np.sqrt(1/np.prod(self.spectral_split))*np.exp(-2*self.xi**2*grid_effect_div_eta)
        
        self.spread_coef = spread_coef
        self.spread_idxs = spread_idxs
        return
        # }}}
    def _contract_precalcs(self):# {{{
        num_points = self.points.shape[0]
        num_contract = num_points*self.offset.shape[0]
        grid_idxs = np.round(self.points/self.grid_spacing).astype(int)
        particle_grid_dist = grid_idxs*self.grid_spacing - self.points
        grid_effect_idxs = (grid_idxs[:,None,:] + self.offset[None,:,:] - 1) % self.num_grid
        contract_idxs = grid_effect_idxs.reshape(num_contract,3)
    
        grid_effect_dist = (particle_grid_dist[:,None,:] + self.offsetxyz[None,:,:])
        grid_effect_div_eta = np.sum(grid_effect_dist**2/self.spectral_split,axis = -1)
        contract_coef = (2*self.xi**2/np.pi)**(3/2)*np.sqrt(1/np.prod(self.spectral_split))*np.exp(-2*self.xi**2*grid_effect_div_eta)

        contract_coef = np.prod(self.grid_spacing)*contract_coef
        contract_coef = contract_coef.reshape(num_contract)
    
        particle_index = np.repeat(np.arange(num_points),len(self.offset))
        E_point = np.zeros([num_points,3]).astype(np.complex128)
        G_point = np.zeros([num_points,5]).astype(np.complex128)

        self.E_point = E_point
        self.G_point = G_point
        self.particle_index = particle_index
        self.contract_coef = contract_coef
        self.contract_idxs = contract_idxs
        return
        # }}}
    def _real_space_precalcs(self):# {{{
        points = self.points
        dip_pos = self.dip_pos

        toQL = np.array([0,0,0,2,3])
        toQR = np.array([0,2,3,2,3])
        I = np.array([1,0,0,1,0])
        self.toQL = toQL
        self.toQR = toQR
        self.I = I

        p1,p2 = self._gen_neighbor_list()
        r = points[p1] - dip_pos[p2]
        r = r-self.box*(2*r/self.box).astype(int)
        d = np.sqrt(np.sum(r**2,axis = -1))
    
        cutoff_flags = d<self.rc
        d = d[cutoff_flags]
        r = r[cutoff_flags]
        r = r[:,:]/d[:,None]
        self.delta = r

        self.__rr = np.array([r[:,0]**2-r[:,2]**2, 2*r[:,0]*r[:,1], 2*r[:,0]*r[:,2], r[:,1]**2-r[:,2]**2, 2*r[:,1]*r[:,2]])
        self.rr = self.__rr.T
    
        self.p1 = p1[cutoff_flags]
        self.p2 = p2[cutoff_flags]
    
        int_perp = interp1d(self.r_table,self.field_dip_1)
        int_para = interp1d(self.r_table,self.field_dip_2)

        int_Q1 = interp1d(self.r_table,self.field_quad_1)
        int_Q2 = interp1d(self.r_table,self.field_quad_2)
        int_Q3 = interp1d(self.r_table,self.field_quad_3)

        int_G1 = interp1d(self.r_table,self.grad_quad_1)
        int_G2 = interp1d(self.r_table,self.grad_quad_2)
        int_G3 = interp1d(self.r_table,self.grad_quad_3)
        int_G4 = interp1d(self.r_table,self.grad_quad_4)
    
        self.self_perp = int_perp(0)
        self.self_G2 = int_G2(0)
        self.perp = int_perp(d)[:,None]
        self.para = int_para(d)[:,None]
        self.Q1 = int_Q1(d)[:,None]
        self.Q2 = int_Q2(d)[:,None]
        self.Q3 = int_Q3(d)[:,None]
        self.G1 = int_G1(d)[:,None]
        self.G2 = int_G2(d)[:,None]
        self.G3 = int_G3(d)[:,None]
        self.G4 = int_G4(d)[:,None]
        # }}}
    def _real_self_precalcs(self):# {{{
        self.self_P_coef = 3/(4*np.pi*(self.eps_p-1))
        self.self_Q_coef = 15/(8*np.pi*(self.eps_p-1))
    # }}}
    def _calc_real_space_table(self):# {{{
    
        r = np.arange(1,10,.001)
        self.r_table = np.insert(r,0,0)
        xi = self.xi
        pi = np.pi
        exp = np.exp
        sqrt = np.sqrt

        #==== Dipole Calcs ===={{{
        # Polynomials multiplying the exponetials
        exppolyp = -(r+2)/(32*pi**(3/2)*xi*r)
        exppolym = -(r-2)/(32*pi**(3/2)*xi*r)
        exppoly0 = 1/(16*pi**(3/2)*xi)
        
        # Polynomials multiplying the error functions
        erfpolyp = (2*xi**2*(r+2)**2 + 1)/(64*pi*xi**2*r)
        erfpolym = (2*xi**2*(r-2)**2 + 1)/(64*pi*xi**2*r)
        erfpoly0 = -(2*xi**2*r**2 + 1)/(32*pi*xi**2*r)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = -1/(4*pi*r) + (4-r)/(16*pi)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        pot_charge = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
        
        ## Potential/Dipole or Field/Charge coupling
        
        # Polynomials multiplying the exponetials
        exppolyp = 1/(256*pi**(3/2)*xi**3*r**2)*(-6*xi**2*r**3 - 4*xi**2*r**2 + (-3+8*xi**2)*r + 2*(1-8*xi**2))
        exppolym = 1/(256*pi**(3/2)*xi**3*r**2)*(-6*xi**2*r**3 + 4*xi**2*r**2 + (-3+8*xi**2)*r - 2*(1-8*xi**2))
        exppoly0 = 3*(2*r**2*xi**2+1)/(128*pi**(3/2)*xi**3*r)
        
        # Polynomials multiplying the error functions
        erfpolyp = 1/(512*pi*xi**4*r**2)*(12*xi**4*r**4 + 32*xi**4*r**3 + 12*xi**2*r**2 - 3+64*xi**4)
        erfpolym = 1/(512*pi*xi**4*r**2)*(12*xi**4*r**4 - 32*xi**4*r**3 + 12*xi**2*r**2 - 3+64*xi**4)
        erfpoly0 = -3*(4*xi**4*r**4 + 4*xi**2*r**2 - 1)/(256*pi*xi**4*r**2)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = -1/(4*pi*r**2) + r/(8*pi)*(1-3/8*r)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        pot_dip = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly


        ## Field/Dipole coupling: I-rr component
    
        # Polynomials multiplying the exponentials
        exppolyp = 1/(1024*pi**(3/2)*xi**5*r**3)*(4*xi**4*r**5 - 8*xi**4*r**4 + 8*xi**2*(2-7*xi**2)*r**3 - \
            8*xi**2*(3+2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
        exppolym = 1/(1024*pi**(3/2)*xi**5*r**3)*(4*xi**4*r**5 + 8*xi**4*r**4 + 8*xi**2*(2-7*xi**2)*r**3 + \
            8*xi**2*(3+2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
        exppoly0 = 1/(512*pi**(3/2)*xi**5*r**2)*(-4*xi**4*r**4 - 8*xi**2*(2-9*xi**2)*r**2 - 3+36*xi**2)
        
        # Polynomials multiplying the error functions
        erfpolyp = 1/(2048*pi*xi**6*r**3)*(-8*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 + 256*xi**6*r**3 - \
            18*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
        erfpolym = 1/(2048*pi*xi**6*r**3)*(-8*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 - 256*xi**6*r**3 - \
            18*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
        erfpoly0 = 1/(1024*pi*xi**6*r**3)*(8*xi**6*r**6 + 36*xi**4*(1-4*xi**2)*r**4 + 18*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = -1/(4*pi*r**3) + 1/(4*pi)*(1-9*r/16+r**3/32)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_dip_1 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
        
        
        ## Field/Dipole coupling: rr component
        
        # Polynomials multiplying the exponentials
        exppolyp = 1/(512*pi**(3/2)*xi**5*r**3)*(8*xi**4*r**5 - 16*xi**4*r**4 + 2*xi**2*(7-20*xi**2)*r**3 - \
            4*xi**2*(3-4*xi**2)*r**2 - (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
        exppolym = 1/(512*pi**(3/2)*xi**5*r**3)*(8*xi**4*r**5 + 16*xi**4*r**4 + 2*xi**2*(7-20*xi**2)*r**3 + \
            4*xi**2*(3-4*xi**2)*r**2 - (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
        exppoly0 = 1/(256*pi**(3/2)*xi**5*r**2)*(-8*xi**4*r**4 - 2*xi**2*(7-36*xi**2)*r**2 + 3-36*xi**2)
        
        # Polynomials multiplying the error functions
        erfpolyp = 1/(1024*pi*xi**6*r**3)*(-16*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 + 128*xi**6*r**3 - 3+36*xi**2-256*xi**6)
        erfpolym = 1/(1024*pi*xi**6*r**3)*(-16*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 - 128*xi**6*r**3 - 3+36*xi**2-256*xi**6)
        erfpoly0 = 1/(512*pi*xi**6*r**3)*(16*xi**6*r**6 + 36*xi**4*(1-4*xi**2)*r**4 + 3-36*xi**2)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = 1/(2*pi*r**3) + 1/(4*pi)*(1-9*r/8+r**3/8)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_dip_2 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
        erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
    
        # Field/Dipole Force: coefficient multiplying -(mi*mj)r and -( (mj*r)mi + (mi*r)mj - 2(mi*r)(mj*r)r )
        # Polynomials multiplying the exponentials
        exppolyp = 3/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 - 8*xi**4*r**4 + 4*xi**2*(1-2*xi**2)*r**3 + 16*xi**4*r**2 - (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
        exppolym = 3/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 + 8*xi**4*r**4 + 4*xi**2*(1-2*xi**2)*r**3 - 16*xi**4*r**2 - (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
        exppoly0 = 3/(512*pi**(3/2)*xi**5*r**3)*(-4*xi**4*r**4 - 4*xi**2*(1-6*xi**2)*r**2 + 3-36*xi**2)
        
        # Polynomials multiplying the error functions
        erfpolyp = 3/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 12*xi**4*(1-4*xi**2)*r**4 + 6*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2-256*xi**6)
        erfpolym = 3/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 12*xi**4*(1-4*xi**2)*r**4 + 6*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2-256*xi**6)
        erfpoly0 = 3/(1024*pi*xi**6*r**4)*(8*xi**6*r**6 + 12*xi**4*(1-4*xi**2)*r**4 - 6*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = 3/(4*pi*r**4) - 3/(64*pi)*(3-r**2/2)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_dip_force_1 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
        
        ## Field/Dipole Force from:  coefficient multiplying -(mi*r)(mj*r)r
        
        # Polynomials multiplying the exponentials
        exppolyp = 9/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 - 8*xi**4*r**4 + 8*xi**4*r**3 + 8*xi**2*(1-2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
        exppolym = 9/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 + 8*xi**4*r**4 + 8*xi**4*r**3 - 8*xi**2*(1-2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
        exppoly0 = 9/(512*pi**(3/2)*xi**5*r**3)*(-4*xi**4*r**4 + 8*xi**4*r**2 - 3+36*xi**2)
        
        # Polynomials multiplying the error functions
        erfpolyp = 9/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 4*xi**4*(1-4*xi**2)*r**4 - 2*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
        erfpolym = 9/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 4*xi**4*(1-4*xi**2)*r**4 - 2*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
        erfpoly0 = 9/(1024*pi*xi**6*r**4)*(8*xi**6*r**6 + 4*xi**4*(1-4*xi**2)*r**4 + 2*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = -9/(4*pi*r**4) - 9/(64*pi)*(1-r**2/2)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_dip_force_2 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
        erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
    
    
        
        # }}}
        #==== Quadrupole Calcs ==== {{{
        #---- potential/quadrupole or field gradient/charge coupling ----{{{
        # polynomials multiplying the exponetials
        exppolyp = 3./(1024*pi**(3/2)*xi**5*r**3)*(20*xi**4*r**5 + 24*xi**4*r**4 + 2*xi**2*(10-4*xi**2)*r**3 \
            - 16*xi**2*(2-xi**2)*r**2 - (15-12*xi**2+32*xi**4)*r + 2*(9-4*xi**2+32*xi**4))
        exppolym = 3./(1024*pi**(3/2)*xi**5*r**3)*(20*xi**4*r**5 - 24*xi**4*r**4 + 2*xi**2*(10-4*xi**2)*r**3 \
            + 16*xi**2*(2-xi**2)*r**2 - (15-12*xi**2+32*xi**4)*r - 2*(9-4*xi**2+32*xi**4))
        exppoly0 = 15/(512*pi**(3/2)*xi**5*r**2)*(4*xi**4*r**4 + 4*xi**2*(1+2*xi**2)*r**2 - 3*(1+4*xi**2))

        # Polynomials multiplying the error functions
        erfpolyp = 3/(2048*pi*xi**6*r**3)*(40*xi**6*r**6 + 128*xi**6*r**5 + 20*xi**4*(3+4*xi**2)*r**4 \
                - 10*xi**2*(3+8*xi**2)*r**2 + 15+60*xi**2+256*xi**6)
        erfpolym = 3/(2048*pi*xi**6*r**3)*(40*xi**6*r**6 - 128*xi**6*r**5 + 20*xi**4*(3+4*xi**2)*r**4 \
                - 10*xi**2*(3+8*xi**2)*r**2 + 15+60*xi**2+256*xi**6)
        erfpoly0 = 15/(1024*pi*xi**6*r**3)*(-8*xi**6*r**6 - 4*xi**4*(3+4*xi**2)*r**4 \
                - 2*xi**2*(3+8*xi**2)*r**2 - 3*(1+4*xi**2))

        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = 15/(1024*pi*xi**6*r**3)*(12*xi**4*r**4 - 2*xi**2*(3+8*xi**2)*r**2 + 3+12*xi**2) # this is wrong; >>> shouldn't have any xi in it
        else:
            regpoly = 0

        # Combine the polynomial coefficients, exponentials, and error functions
        pot_quad = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) \
                + exppoly0*exp(-r**2*xi**2) + erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) \
                + erfpoly0*erfc(r*xi) + (r < 2)*regpoly;
        # }}}
        #---- Field/Quadrupole or Field Gradient/Dipole Coupling: rrr*S component ----{{{

        # Polynomials multiplying the exponentials
        exppolyp = 15/(16384*pi**(3/2)*xi**7*r**4)*( -24*xi**6*r**7 + 48*xi**6*r**6 - 4*xi**4*(9-8*xi**2)*r**5 + 8*xi**4*(3-8*xi**2)*r**4 \
            + 2*xi**2*(21-80*xi**2+64*xi**4)*r**3 - 4*xi**2*(3-8*xi**2)**2*r**2 - (45-120*xi**2+192*xi**4-512*xi**6)*r - 2*(45+24*xi**2-64*xi**4+512*xi**6) );
        exppolym = 15/(16384*pi**(3/2)*xi**7*r**4)*( -24*xi**6*r**7 - 48*xi**6*r**6 - 4*xi**4*(9-8*xi**2)*r**5 - 8*xi**4*(3-8*xi**2)*r**4 \
            + 2*xi**2*(21-80*xi**2+64*xi**4)*r**3 + 4*xi**2*(3-8*xi**2)**2*r**2 - (45-120*xi**2+192*xi**4-512*xi**6)*r + 2*(45+24*xi**2-64*xi**4+512*xi**6) );
        exppoly0 = 15/(8192*pi**(3/2)*xi**7*r**3)*( 24*xi**6*r**6 + 4*xi**4*(9-32*xi**2)*r**4 - 2*xi**2*(21-128*xi**2)*r**2 + 45-480*xi**2 );
        
        # Polynomials multiplying the error functions
        erfpolyp = 15/(32768*pi*xi**8*r**4)*( 48*xi**8*r**8 + 32*xi**6*(3-8*xi**2)*r**6 - 24*xi**4*(3-16*xi**2)*r**4 \
            + 72*xi**2*(1-8*xi**2)*r**2 -45+480*xi**2+4096*xi**8 );
        erfpolym = 15/(32768*pi*xi**8*r**4)*( 48*xi**8*r**8 + 32*xi**6*(3-8*xi**2)*r**6 - 24*xi**4*(3-16*xi**2)*r**4 \
            + 72*xi**2*(1-8*xi**2)*r**2 -45+480*xi**2+4096*xi**8 );
        erfpoly0 = 15/(16384*pi*xi**8*r**4)*( -48*xi**8*r**8 - 32*xi**6*(3-8*xi**2)*r**6 + 24*xi**4*(3-16*xi**2)*r**4 \
            - 72*xi**2*(1-8*xi**2)*r**2 +45-480*xi**2 );
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = -15/(4*pi*r**4) + 15*r**2/(64*pi)*(1-3*r**2/16);
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_quad_1 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly;
        # }}}
        #---- Field/Quadrupole or Field Gradient/Dipole Coupling: Ir*S+Sr+rS component ----{{{
        
        # Polynomials multiplying the exponentials
        exppolyp = 3/(16384*pi**(3/2)*xi**7*r**4)*( -40*xi**6*r**7 + 80*xi**6*r**6 - 20*xi**4*(11-24*xi**2)*r**5 + 8*xi**4*(45+8*xi**2)*r**4 \
            - 2*xi**2*(45-80*xi**2+64*xi**4)*r**3 - 4*xi**2*(15+48*xi**2-64*xi**4)*r**2 + (45-120*xi**2+192*xi**4-512*xi**6)*r + 2*(45+24*xi**2-64*xi**4+512*xi**6) );
        exppolym = 3/(16384*pi**(3/2)*xi**7*r**4)*( -40*xi**6*r**7 - 80*xi**6*r**6 - 20*xi**4*(11-24*xi**2)*r**5 - 8*xi**4*(45+8*xi**2)*r**4 \
            - 2*xi**2*(45-80*xi**2+64*xi**4)*r**3 + 4*xi**2*(15+48*xi**2-64*xi**4)*r**2 + (45-120*xi**2+192*xi**4-512*xi**6)*r - 2*(45+24*xi**2-64*xi**4+512*xi**6) );
        exppoly0 = 15/(8192*pi**(3/2)*xi**7*r**3)*( 8*xi**6*r**6 + 4*xi**4*(11-32*xi**2)*r**4 + 2*xi**2*(9-64*xi**2)*r**2 -9+96*xi**2 );
        
        # Polynomials multiplying the error functions
        erfpolyp = 3/(32768*pi*xi**8*r**4)*( 80*xi**8*r**8 + 160*xi**6*(3-8*xi**2)*r**6 - 2048*xi**8*r**5 + 120*xi**4*(3-16*xi**2)*r**4 \
            - 120*xi**2*(1-8*xi**2)*r**2 + 45-480*xi**2-4096*xi**8 );
        erfpolym = 3/(32768*pi*xi**8*r**4)*( 80*xi**8*r**8 + 160*xi**6*(3-8*xi**2)*r**6 + 2048*xi**8*r**5 + 120*xi**4*(3-16*xi**2)*r**4 \
            - 120*xi**2*(1-8*xi**2)*r**2 + 45-480*xi**2-4096*xi**8 );
        erfpoly0 = 15/(16384*pi*xi**8*r**4)*( -16*xi**8*r**8 - 32*xi**6*(3-8*xi**2)*r**6 - 24*xi**4*(3-16*xi**2)*r**4 + 24*xi**2*(1-8*xi**2)*r**2 -9+96*xi**2);
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = 3/(4*pi*r**4) - 3*r/(8*pi)*(1-5*r/8+5*r**3/128);
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_quad_2 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly; 
        # }}}
        #---- Field/Quadrupole or Field Gradient/Dipole Coupling: Ir*S component ---- {{{

        # Polynomials multiplying the exponentials
        exppolyp = 5/(1024*pi**(3/2)*xi**5*r**2)*( 4*xi**4*r**5 - 8*xi**4*r**4 + 16*xi**2*(1-2*xi**2)*r**3 - 24*xi**2*r**2 + 3*r + 6);
        exppolym = 5/(1024*pi**(3/2)*xi**5*r**2)*( 4*xi**4*r**5 + 8*xi**4*r**4 + 16*xi**2*(1-2*xi**2)*r**3 + 24*xi**2*r**2 + 3*r - 6);
        exppoly0 = 5/(512*pi**(3/2)*xi**5*r)*( -4*xi**4*r**4 - 16*xi**2*(1-3*xi**2)*r**2 -3+24*xi**2 );
        
        # Polynomials multiplying the error functions
        erfpolyp = 5/(2048*pi*xi**6*r**2)*( -8*xi**6*r**6 - 12*xi**4*(3-8*xi**2)*r**4 + 128*xi**6*r**3 - 6*xi**2*(3-16*xi**2)*r**2 + 3-24*xi**2 );
        erfpolym = 5/(2048*pi*xi**6*r**2)*( -8*xi**6*r**6 - 12*xi**4*(3-8*xi**2)*r**4 - 128*xi**6*r**3 - 6*xi**2*(3-16*xi**2)*r**2 + 3-24*xi**2 );
        erfpoly0 = 5/(1024*pi*xi**6*r**2)*( 8*xi**6*r**6 + 12*xi**4*(3-8*xi**2)*r**4 + 6*xi**2*(3-16*xi**2)*r**2 -3+24*xi**2 );
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = 5*r/(8*pi)*(1-3*r/4+r**3/16);  
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_quad_3 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly;
            
        #}}}
        #---- Field Gradient/Quadrupole Coupling: Irr:Q component ----{{{
        
        # Polynomials multiplying the exponentials
        exppolyp = 75/(16384*pi**(3/2)*xi**7*r**3)*( 8*xi**6*r**7 - 16*xi**6*r**6 + (44*xi**4-32*xi**6)*r**5 - (72*xi**4-64*xi**6)*r**4 \
            + (18*xi**2+32*xi**4)*r**3 + (12*xi**2-64*xi**4)*r**2 - (9+24*xi**2)*r - (18-48*xi**2) )
        exppolym = 75/(16384*pi**(3/2)*xi**7*r**3)*( 8*xi**6*r**7 + 16*xi**6*r**6 + (44*xi**4-32*xi**6)*r**5 + (72*xi**4-64*xi**6)*r**4 \
            + (18*xi**2+32*xi**4)*r**3 - (12*xi**2-64*xi**4)*r**2 - (9+24*xi**2)*r + (18-48*xi**2) )
        exppoly0 = -75/(8192*pi**(3/2)*xi**7*r**2)*( 8*xi**6*r**6 + (44*xi**4-64*xi**6)*r**4 + (18*xi**2-64*xi**4+128*xi**6)*r**2 -9+48*xi**2-192*xi**4 )
        
        # Polynomials multiplying the error functions
        erfpolyp = -75/(32768*pi*xi**8*r**3)*( 16*xi**8*r**8 + (96*xi**6-128*xi**8)*r**6 + (72*xi**4-192*xi**6+256*xi**8)*r**4 \
            - (24*xi**2-96*xi**4+256*xi**6)*r**2 +9-48*xi**2+192*xi**4 )
        erfpolym = -75/(32768*pi*xi**8*r**3)*( 16*xi**8*r**8 + (96*xi**6-128*xi**8)*r**6 + (72*xi**4-192*xi**6+256*xi**8)*r**4 \
            - (24*xi**2-96*xi**4+256*xi**6)*r**2 +9-48*xi**2+192*xi**4 )
        erfpoly0 = 75/(16384*pi*xi**8*r**3)*( 16*xi**8*r**8 + (96*xi**6-128*xi**8)*r**6 + (72*xi**4-192*xi**6+256*xi**8)*r**4 \
            - (24*xi**2-96*xi**4+256*xi**6)*r**2 +9-48*xi**2+192*xi**4 )
                                                                                                                                                          
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = 75*r*(4-r**2)**2/(1024*pi)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        grad_quad_1 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly;

        #}}}
        #---- Field Gradient/Quadrupole Coupling: Q component ----{{{
        
        # Polynomials multiplying the exponentials
        exppolyp = 3/(32768*pi**(3/2)*xi**9*r**5)*( -48*xi**8*r**9 + 96*xi**8*r**8 - (576*xi**6-608*xi**8)*r**7 + (1056*xi**6-1216*xi**8)*r**6 \
            - (1536*xi**4-2576*xi**6+3968*xi**8)*r**5 + (2160*xi**4-4320*xi**6-256*xi**8)*r**4 - (360*xi**2+360*xi**4+640*xi**6-512*xi**8)*r**3 \
            - (360*xi**2-1680*xi**4-768*xi**6+1024*xi**8)*r**2 + (135+180*xi**2+480*xi**4-768*xi**6+2048*xi**8)*r + (270-1080*xi**2-192*xi**4+512*xi**6-4096*xi**8) )
        exppolym = 3/(32768*pi**(3/2)*xi**9*r**5)*( -48*xi**8*r**9 - 96*xi**8*r**8 - (576*xi**6-608*xi**8)*r**7 - (1056*xi**6-1216*xi**8)*r**6 \
            - (1536*xi**4-2576*xi**6+3968*xi**8)*r**5 - (2160*xi**4-4320*xi**6-256*xi**8)*r**4 - (360*xi**2+360*xi**4+640*xi**6-512*xi**8)*r**3 \
            + (360*xi**2-1680*xi**4-768*xi**6+1024*xi**8)*r**2 + (135+180*xi**2+480*xi**4-768*xi**6+2048*xi**8)*r - (270-1080*xi**2-192*xi**4+512*xi**6-4096*xi**8) )
        exppoly0 = 1/(16384*pi**(3/2)*xi**9*r**4)*( 144*xi**8*r**8 + (1728*xi**6-2400*xi**8)*r**6 + (4608*xi**4-13200*xi**6+19200*xi**8)*r**4 \
            + (1080*xi**2-5400*xi**4+19200*xi**6)*r**2 -405+2700*xi**2-14400*xi**4 )
        
        # Polynomials multiplying the error functions
        erfpolyp = 1/(65536*pi*xi**10*r**5)*( 288*xi**10*r**10 + (3600*xi**8-4800*xi**10)*r**8 + (10800*xi**6-28800*xi**8+38400*xi**10)*r**6 \
            + 49152*xi**10*r**5 + (5400*xi**4-21600*xi**6+57600*xi**8)*r**4 - (1350*xi**2-7200*xi**4+28800*xi**6)*r**2 +405-2700*xi**2+14400*xi**4+49152*xi**10 )
        erfpolym = 1/(65536*pi*xi**10*r**5)*( 288*xi**10*r**10 + (3600*xi**8-4800*xi**10)*r**8 + (10800*xi**6-28800*xi**8+38400*xi**10)*r**6 \
            - 49152*xi**10*r**5 + (5400*xi**4-21600*xi**6+57600*xi**8)*r**4 - (1350*xi**2-7200*xi**4+28800*xi**6)*r**2 +405-2700*xi**2+14400*xi**4+49152*xi**10 )
        erfpoly0 = -1/(32768*pi*xi**10*r**5)*( 288*xi**10*r**10 + (3600*xi**8-4800*xi**10)*r**8 + (10800*xi**6-28800*xi**8+38400*xi**10)*r**6 \
            + (5400*xi**4-21600*xi**6+57600*xi**8)*r**4 - (1350*xi**2-7200*xi**4+28800*xi**6)*r**2 +405-2700*xi**2+14400*xi**4 )
        
        # Regularization for overlapping particles
        regpoly = -3/(2*pi)*(1/r**5-1+25*r/32-25*r**3/256+3*r**5/512)
        
        # Combine the polynomial coefficients, exponentials, and error functions
        grad_quad_2 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
        
        #}}}
        #---- Field Gradient/Quadrupole Coupling: Irr:Q+2Q*rr+2rr*Q component ----{{{
        
        # Polynomials multiplying the exponentials
        exppolyp = -15/(65536*pi**(3/2)*xi**9*r**5)*( 48*xi**8*r**9 - 96*xi**8*r**8 + (336*xi**6-288*xi**8)*r**7 - 576*(xi**6-xi**8)*r**6 \
            + (216*xi**4+144*xi**6+128*xi**8)*r**5 - (480*xi**6+256*xi**8)*r**4 - (180*xi**2-120*xi**4+640*xi**6-512*xi**8)*r**3 \
            + (720*xi**4+768*xi**6-1024*xi**8)*r**2 + (135+180*xi**2+480*xi**4-768*xi**6+2048*xi**8)*r - (-270+1080*xi**2+192*xi**4-512*xi**6+4096*xi**8) )
        exppolym = -15/(65536*pi**(3/2)*xi**9*r**5)*( 48*xi**8*r**9 + 96*xi**8*r**8 + (336*xi**6-288*xi**8)*r**7 + 576*(xi**6-xi**8)*r**6 \
            + (216*xi**4+144*xi**6+128*xi**8)*r**5 + (480*xi**6+256*xi**8)*r**4 - (180*xi**2-120*xi**4+640*xi**6-512*xi**8)*r**3 \
            - (720*xi**4+768*xi**6-1024*xi**8)*r**2 + (135+180*xi**2+480*xi**4-768*xi**6+2048*xi**8)*r + (-270+1080*xi**2+192*xi**4-512*xi**6+4096*xi**8) )
        exppoly0 = 15/(32768*pi**(3/2)*xi**9*r**4)*( 48*xi**8*r**8 + (336*xi**6-480*xi**8)*r**6 + (216*xi**4-720*xi**6+1280*xi**8)*r**4 \
            - (180*xi**2-840*xi**4+2560*xi**6)*r**2 +135-900*xi**2+4800*xi**4 )
        
        # Polynomials multiplying the error functions
        erfpolyp = -15/(131072*pi*xi**10*r**5)*( -96*xi**10*r**10 + (-720*xi**8+960*xi**10)*r**8 - (720*xi**6-1920*xi**8+2560*xi**10)*r**6 \
            + (360*xi**4-1440*xi**6+3840*xi**8)*r**4 - (270*xi**2-1440*xi**4+5760*xi**6)*r**2 +135-900*xi**2+4800*xi**4+16384*xi**10 )
        erfpolym = -15/(131072*pi*xi**10*r**5)*( -96*xi**10*r**10 + (-720*xi**8+960*xi**10)*r**8 - (720*xi**6-1920*xi**8+2560*xi**10)*r**6 \
            + (360*xi**4-1440*xi**6+3840*xi**8)*r**4 - (270*xi**2-1440*xi**4+5760*xi**6)*r**2 +135-900*xi**2+4800*xi**4+16384*xi**10 )
        erfpoly0 = -15/(65536*pi*xi**10*r**5)*( 96*xi**10*r**10 + (720*xi**8-960*xi**10)*r**8 + (720*xi**6-1920*xi**8+2560*xi**10)*r**6 \
            - (360*xi**4-1440*xi**6+3840*xi**8)*r**4 + (270*xi**2-1440*xi**4+5760*xi**6)*r**2 -135+900*xi**2-4800*xi**4 )
        
        # Regularization for overlapping particles
        regpoly = -15*(r**2-4)**3*(3*r**4+6*r**2+8)/(2048*pi*r**5)
        
        # Combine the polynomial coefficients, exponentials, and error functions                                                                                
        grad_quad_3 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
        # }}}
        #---- Field Gradient/Quadrupole Coupling: rrrr:Q component ----{{{
        
        # Polynomials multiplying the exponentials
        exppolyp = -15/(65536*pi**(3/2)*xi**9*r**5)*( 144*xi**8*r**9 - 288*xi**8*r**8 + (288*xi**6+96*xi**8)*r**7 - (288*xi**6+192*xi**8)*r**6 \
            - (432*xi**4-912*xi**6+896*xi**8)*r**5 + (720*xi**4-480*xi**6+1792*xi**8)*r**4 + (720*xi**2-2280*xi**4+4480*xi**6-3584*xi**8)*r**3 \
            - (1080*xi**2+2160*xi**4+5376*xi**6-7168*xi**8)*r**2 - (945+1260*xi**2+3360*xi**4-5376*xi**6+14336*xi**8)*r \
            - (1890-7560*xi**2-1344*xi**4+3584*xi**6-28672*xi**8) )
        exppolym = -15/(65536*pi**(3/2)*xi**9*r**5)*( 144*xi**8*r**9 + 288*xi**8*r**8 + (288*xi**6+96*xi**8)*r**7 + (288*xi**6+192*xi**8)*r**6 \
            - (432*xi**4-912*xi**6+896*xi**8)*r**5 - (720*xi**4-480*xi**6+1792*xi**8)*r**4 + (720*xi**2-2280*xi**4+4480*xi**6-3584*xi**8)*r**3 \
            + (1080*xi**2+2160*xi**4+5376*xi**6-7168*xi**8)*r**2 - (945+1260*xi**2+3360*xi**4-5376*xi**6+14336*xi**8)*r \
            + (1890-7560*xi**2-1344*xi**4+3584*xi**6-28672*xi**8) )
        exppoly0 = 15/(32768*pi**(3/2)*xi**9*r**4)*( 144*xi**8*r**8 + (288*xi**6-480*xi**8)*r**6 - (432*xi**4-1200*xi**6+1280*xi**8)*r**4 \
            + (720*xi**2-3000*xi**4+6400*xi**6)*r**2 -945+6300*xi**2-33600*xi**4 )
        
        # Polynomials multiplying the error functions
        erfpolyp = 15/(131072*pi*xi**10*r**5)*( 288*xi**10*r**10 + (720*xi**8-960*xi**10)*r**8 - (720*xi**6-1920*xi**8+2560*xi**10)*r**6 \
            + (1080*xi**4-4320*xi**6+11520*xi**8)*r**4 - (1350*xi**2-7200*xi**4+28800*xi**6)*r**2 +945-6300*xi**2+33600*xi**4+114688*xi**10 )
        erfpolym = 15/(131072*pi*xi**10*r**5)*( 288*xi**10*r**10 + (720*xi**8-960*xi**10)*r**8 - (720*xi**6-1920*xi**8+2560*xi**10)*r**6 \
            + (1080*xi**4-4320*xi**6+11520*xi**8)*r**4 - (1350*xi**2-7200*xi**4+28800*xi**6)*r**2 +945-6300*xi**2+33600*xi**4+114688*xi**10 )
        erfpoly0 = -15/(65536*pi*xi**10*r**5)*( 288*xi**10*r**10 + (720*xi**8-960*xi**10)*r**8 - (720*xi**6-1920*xi**8+2560*xi**10)*r**6 \
            + (1080*xi**4-4320*xi**6+11520*xi**8)*r**4 - (1350*xi**2-7200*xi**4+28800*xi**6)*r**2 +945-6300*xi**2+33600*xi**4 )
        
        # Regularization for overlapping particles
        regpoly = -15*(3584 - 80*r**6 - 30*r**8 + 9*r**10)/(2048*pi*r**5)
        
        # Combine the polynomial coefficients, exponentials, and error functions
        grad_quad_4 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
        # }}}
        #}}}
        #==== Self terms ===={{{
        # Potential/charge
        selfo = (1-exp(-4*xi**2))/(8*pi**(3/2)*xi) + erfc(2*xi)/(4*pi)
        pot_charge = np.insert(pot_charge,0,selfo)
        
        # Potential/dipole or field/charge
        pot_dip = np.insert(pot_dip,0,0)
        
        # Field/dipole
        selfo = (-1+6*xi**2+(1-2*xi**2)*exp(-4*xi**2))/(16*pi**(3/2)*xi**3) + erfc(2*xi)/(4*pi)
        self.field_dip_1 = np.insert(field_dip_1,0,selfo)
        self.field_dip_2 = np.insert(field_dip_2,0,selfo)
        
        # Field/dipole force
        field_dip_force_1 = np.insert(field_dip_force_1,0,0)
        field_dip_force_2 = np.insert(field_dip_force_2,0,0)

        self.field_quad_1 = np.insert(field_quad_1,0,0)
        self.field_quad_2 = np.insert(field_quad_2,0,0)
        self.field_quad_3 = np.insert(field_quad_3,0,0)

        selfo = 1/(2*pi)*( 3*(3-10*xi**2+20*xi**4)/(8*sqrt(pi)*xi**5) - 3*(3+2*xi**2+4*xi**4)*exp(-4*xi**2)/(8*sqrt(pi)*xi**5) + 3*erfc(2*xi) )
        self.grad_quad_1 = np.insert(grad_quad_1,0,selfo)
        self.grad_quad_2 = np.insert(grad_quad_2,0,selfo)
        self.grad_quad_3 = np.insert(grad_quad_3,0,selfo)
        self.grad_quad_4 = np.insert(grad_quad_4,0,selfo)


        #}}}
        return
    # }}}
    def _precalculations(self):# {{{
        self.rc = np.sqrt(-np.log(self.errortol))/self.xi
        if np.any(self.rc > self.box/2):
            raise Exception(f"Real space cutoff ({self.rc:.3f}) larger than half the box length.")

        self._calc_real_space_table()

        self.kcut = 2*self.xi**2*self.rc
        self.num_grid = np.ceil(1+self.box*self.kcut/np.pi).astype(int)
        self.grid_spacing = self.box/self.num_grid
        self.num_grid_gaussian = np.ceil(-2*np.log(self.errortol)/np.pi)
        self.spectral_split = self.num_grid_gaussian * (self.grid_spacing*self.xi)**2/np.pi
        self.E_grid = np.zeros(np.append(self.num_grid,3)).astype(np.complex128)
        self.G_grid = np.zeros(np.append(self.num_grid,5)).astype(np.complex128)

        off = int(self.num_grid_gaussian/2)
        min_off = -off
        max_off = off+1
        offset = []
        for x in range(min_off,max_off):
            for y in range(min_off,max_off):
                for z in range(min_off,max_off):
                    offset.append([x,y,z])
        self.offset = np.array(offset)[:,[2,1,0]]
        self.offsetxyz = self.offset*self.grid_spacing

        #---- Scale Precalcs ----{{{
        warnings.filterwarnings('ignore')
    
        Kx = np.arange(-np.ceil((self.num_grid[0]-1)/2),np.floor((self.num_grid[0] - 1)/2)+1) * 2*np.pi/self.box[0]
        Ky = np.arange(-np.ceil((self.num_grid[1]-1)/2),np.floor((self.num_grid[1] - 1)/2)+1) * 2*np.pi/self.box[1]
        Kz = np.arange(-np.ceil((self.num_grid[2]-1)/2),np.floor((self.num_grid[2] - 1)/2)+1) * 2*np.pi/self.box[2]
    
        k0x = np.argwhere(Kx == 0)
        k0y = np.argwhere(Ky == 0)
        k0z = np.argwhere(Kz == 0)
        k0_ind = np.array([k0x,k0y,k0z])
    
        kx,ky,kz = np.meshgrid(Kx,Ky,Kz,indexing='ij')
        k = np.concatenate([kx[:,:,:,None],ky[:,:,:,None],kz[:,:,:,None]],axis = -1)
    
        ksq = k**2
        ksqsm = np.sum(ksq,axis = -1)
        kmag = np.sqrt(ksqsm)
        khat = k/kmag[:,:,:,None]
        khat[*k0_ind] = 0

        self.Qfactor = np.concatenate([(khat[...,0]*khat[...,0] - 1/3)[...,None],
                                  (khat[...,0]*khat[...,1])[...,None],
                                  (khat[...,0]*khat[...,2])[...,None],
                                  (khat[...,1]*khat[...,1] - 1/3)[...,None],
                                  (khat[...,1]*khat[...,2])[...,None]],axis=-1)

        self.Qfactor_dot = np.concatenate([(khat[...,0]*khat[...,0] - khat[...,2]**2)[...,None],
                                   2*(khat[...,0]*khat[...,1])[...,None],
                                   2*(khat[...,0]*khat[...,2])[...,None],
                                     (khat[...,1]*khat[...,1] - khat[...,2]**2)[...,None],
                                   2*(khat[...,1]*khat[...,2])[...,None]],axis=-1)

        etaksq  = np.sum(ksq*(1-self.spectral_split),axis = -1)
        expksq = np.exp(-etaksq/(4*self.xi**2)) / ksqsm
        expk2 = expksq
        j1 = np.sqrt(np.pi/(2*kmag)) * besselj(1.5,kmag)
        j2 = np.sqrt(np.pi/(2*kmag)) * besselj(2.5,kmag)

        scale_coef = 9 * j1**2 * expk2
        scale_coef[*k0_ind] = 0
        self.scale_EP_coef = scale_coef

        scale_coef = -45/2*1j*j1*j2*expk2
        scale_coef[*k0_ind] = 0
        self.scale_EQ_coef = scale_coef

        scale_coef = 45*1j*j2*j1*expk2
        scale_coef[*k0_ind] = 0
        self.scale_GP_coef = scale_coef

        scale_coef = 225/2*j2**2*expk2
        scale_coef[*k0_ind] = 0
        self.scale_GQ_coef = scale_coef


        self.khat = khat
        warnings.filterwarnings('default')
        # }}}
        return 
    # }}}
    def _gen_neighbor_list(self,):#{{{
        box_length = self.box
        cutoff = self.rc
        points = self.points
        dip_pos = self.dip_pos

        if np.any(points < 0) or np.any(dip_pos < 0):
            points = np.copy(points)
            points += box_length/2

            dip_pos = np.copy(dip_pos)
            dip_pos += box_length/2
        numBoxes = (box_length/cutoff).astype(int)


        cutoff = box_length/numBoxes
        if len(points.shape) == 2:
            numFrames = 1
            numPoints = points.shape[0]
            numDips = dip_pos.shape[0]
            dims = points.shape[1]
        elif len(points.shape) == 3:
            numFrames = points.shape[0]
            numPoints = points.shape[1]
            numDips = dip_pos.shape[0]
            dims = points.shape[2]

        flag = numBoxes <= 3
        if np.all(flag):
            point_idx = np.arange(numPoints)
            dip_idx = np.arange(numDips)
            p1 = np.repeat(point_idx,numDips)
            p2 = np.tile(dip_idx,numPoints)
            flag = p1 != p2
            return p1[flag],p2[flag]
        elif np.any(flag):
            points = points[:,~flag]
            dip_pos = dip_pos[:,~flag]
            numBoxes = numBoxes[~flag]
            cutoff = cutoff[~flag]
            dims = np.count_nonzero(~flag)

        point_indices = (points/cutoff).astype(int)
        if not np.iterable(numBoxes):
            numBoxes = np.array([numBoxes]*dims)
        boxes = {}
        count = np.zeros(numBoxes)
        for i,idx in enumerate(point_indices):
            index = tuple(idx.tolist())
            if index not in boxes:
                boxes[index] = list()
            boxes[index].append(int(i))
            count[index] += 1
        P1,P2 = [],[]
        offset = np.array([np.arange(-1,2) for i in range(dims)]).T

        dip_pos_indices = (dip_pos/cutoff).astype(int)
        for p1,idx in enumerate(dip_pos_indices):
            offsets = (idx+offset)%numBoxes
            for off in itertools.product(*offsets.T):
                if count[off] == 0:
                    continue
                p2s = boxes[off]
                P1 += [p1]*len(p2s)
                P2 += (p2s)
        P1 = np.array(P1)
        P2 = np.array(P2)
        if self.calc_inter_dipole:
            flags = P1 != P2
            P1 = P1[flags]
            P2 = P2[flags]
        return P2,P1
        #}}}
# }}}
