# L3_internship_Francois_Illan
The following file gathers the functions I made during my L3 Internship at the Lumin/Jean Perrin labs. I analyzed the kinetic of some signals produced by organoïds shot by a laser beam. I also studied the propagation of these signals among the organoïd and the theorical temperature profile in the organoïd.

If you want to get the kinetic of the excited points, first use eclairement_moyen, and then acquire_pts_excites. Then you will be able to use affichage_rise_decay_all_runs et affichage_rise_decay_same_parameters (you can also use the function tracer_allure_moyenne to get the mean signal).

In order to study the propagationb of the signal, first execute acquire_zone_cartographie, then you can execute propagation_activation_complete and visualisation_grille_temporelle_interactive (I think that these algorithms need improvement)

Finally, in order to study theorically the temperature in the organoïd, you can use profil_temperature.
