#------------------------------------------------------------------------------
# Definition of format strings
#------------------------------------------------------------------------------
f222 = " {:12.5f} {:12.5f}\n"
f332 = " {:4d}{:2d} {:12.5f} {:12.5f}\n"
f333 = " {:12.5f} {:12.5f} {:12.5f}\n"
f443 = " {:4d} {:12.5f} {:12.5f} {:12.5f}\n"
f444 = " {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n"
f551 = "  {:1d} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n"
f555 = " {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n"
f556 = " {:4d} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n"
f666 = " {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n"
f777 = " {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n"
f101 = " f    = {:8.2f} n     = {:5d} a   = {:5.4f}\n"
f102 = " nmc  = {:8d} neq   = {:5d}\n"
f1102= " nin  = {:8d} nmc   = {:5d} neq = {:5d}\n"
f103 = " np   = {:8d} nc    = {:5d}\n" 
f104 = " delx = {:8.2f} icold = {:5d}\n"
f1104= "dz = {:8.2f} tcor= {:5.3f} scor= {:8.2f}"
f105 = " w_0  = {:8.2f} nalp = {:5d}\n"
f106 = " delx = {:8.2f} nalp = {:5d}\n"
f107 = " delx = {:8.2f} nheat = {:5d}\n"
f201 = " f    = {:8.2f} n    = {:8d} a   = {:8.4f}\n"
f202 = " nmc  = {:8d} neq  = {:8d}\n"
f203 = " np   = {:8d} nc   = {:8d}\n"
f204 = " delx = {:8.2f} icold= {:8d} ncool= {:8d}\n"
f205 = " S_0  = {:8.2f} dE   = {:8.2f} dE*L = {:8.2f}\n"
f206 = " S_0  = {:8.2f} dE_2 = {:8.2f} dE2*L={:8.2f}\n"
f301 = " f = {:8.5f} s0(num) = {:8.2f} s(ana) = {:8.2f}\n"
f601 = " f   = {:12.4f} ndim = {:5d}\n"
f602 = " t_m = {:12.4f} ntau = {:5d}\n"
f603 = " x_m = {:12.4f} nx   = {:5d}\n"
f800 = " alpha  = {:12.5f}\n"
f801 = " stot   = {:12.5f} ± {:12.5f}\n"
f802 = " v      = {:12.5f} ± {:12.5f}\n"
f803 = " t_av   = {:12.5f} ± {:12.5f}\n"
f804 = " t(vir) = {:12.5f} ± {:12.5f}\n"
f805 = " e_aV   = {:12.5f} ± {:12.5f}\n"
f806 = " x      = {:12.5f} ± {:12.5f}\n"
f807 = " x^2    = {:12.5f} ± {:12.5f}\n"
f808 = " x^4    = {:12.5f} ± {:12.5f}\n"
f809 = " v_alp  = {:12.5f} ± {:12.5f}\n"
f810 = " E_i    = {:12.5f} dE = {:12.5f} E_0 = {:12.5f}\n"
f811 = " beta   = {:12.5f} F0 = {:12.5f} E_0 = {:12.5f}\n"
f812 = " E_i    = {:12.5f} ± {:12.5f}\n"
f813 = " stat   = {:12.5f} up/d {:12.5f} disc= {:12.5f}\n"
f814 = " S_ng   = {:12.5f} dS = {:12.5f}\n"
f1814= " Svac_n = {:12.5f} ± {:12.5f}\n"
f815 = " S_ng   = {:12.5f} ± {:12.5f}\n"
f1816= " Fvac_n = {:12.5f} ± {:12.5f}\n"
f817 = " seff = {:12.5f} ins = {:12.5f} vac = {:12.5f}\n"
f818 = " seff = {:12.5f} ± {:12.5f}\n"
f819 = " dens = {:12.5f} gaus = {:12.5f}\n"
f820 = " dens = {:12.5f} ± {:12.5f}\n"
f901 = "  n       E_n         |c_n|^2\n"
f902 = "         x          psi(x)     psi(x)^2\n"
f903 = "         t         x(0)x(t)     1 state     3 states\n"
f904 = "         t      log x(0)x(t)    1 state     3 states\n"
f905 = "         t      d log(pi)/dt    1 state     3 states\n"
f906 = "         t         x(0)x(t)\n"
f907 = "         t      x^2(0)x^2(t)    1 state     3 states\n"
f908 = "   t            x^2(0)x^2(t)\n"
f909 = "         t      x^3(0)x^3(t)    1 state     3 states\n"
f910 = "   t            x^3(0)x^3(t)\n"
f9901 = " stot   = {:12.5f} ± {:12.5f}\n"
f9902 = " s/nin  = {:12.5f} ± {:12.5f}\n"
f9903 = " s0     = {:12.5f}\n"
f9904 = " si/s0  = {:12.5f} ± {:12.5f}\n"
f9905 = " v_av   = {:12.5f} ± {:12.5f}\n"
f9906 = " t_av   = {:12.5f} ± {:12.5f}\n"
f9907 = " t(vir) = {:12.5f} ± {:12.5f}\n"
f9908 = " e_av   = {:12.5f} ± {:12.5f}\n"
f9909 = " x      = {:12.5f} ± {:12.5f}\n"
f9910 = " x^2    = {:12.5f} ± {:12.5f}\n"
f9911 = " x^4    = {:12.5f} ± {:12.5f}"