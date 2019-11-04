import numpy as np

# dictionary of the optimal values of the several test cases that we have run so
# far. The description of the keys have been provided in comments above the keys
sc_sol_dict = { 'sc_init' : {'twist_cp' : np.array([2.5, 2.5, 5.0]), # 2.5*np.ones(3),
                             'thickness_cp' : np.array([0.008, 0.008, 0.008]), # 1.e-3 * np.array([5.5, 5.5, 5.5]),
                             'sweep' : 20.,
                             'alpha' : 5.,
                            },

                # E = 25GPa
                # 'deterministic_lf1' : {'twist_cp' : np.array([3.37718983, 10, 5]) ,
                #                        'thickness_cp' : np.array([0.001, 0.001, 0.00114519]), # 1.e-3 * np.ones(3) ,
                #                        'sweep' : 17.97227386,
                #                        'alpha' : -0.24701157},

                'deterministic_lf1' : {'twist_cp' : np.array([4.11604326, 10, 5]) ,
                                       'thickness_cp' : np.array([0.001, 0.001, 0.001307917]), # 1.e-3 * np.ones(3) ,
                                       'sweep' : 18.09811024,
                                       'alpha' : 0.58689839},

                # E = 25GPa
                # 'deterministic_lf2.5' : {'twist_cp' : np.array([10., 10, 5]) ,
                #                          'thickness_cp' : np.array([0.001, 0.001, 0.002222788]),
                #                          'sweep' : 19.07479829,
                #                          'alpha' : 10.},
                'deterministic_lf2.5' : {'twist_cp' : np.array([10., 10, 5]) ,
                                         'thickness_cp' : np.array([0.001, 0.001, 0.00229426]),
                                         'sweep' : 18.90680765,
                                         'alpha' : 10.},


                # Optimal design values obtained using the initial deterministic design variables
                # Seed: Initial design, 7 RV, sample radius = 1.e-1, n_dominant_dir = 1, rdo factor=2
                'init_7rv_1e_1_1_2' :  {'twist_cp' : np.array([4.5766595, 10, 5]),
                                   'thickness_cp' : 1.e-3 * np.ones(3),
                                   'sweep' : 17.93557251,
                                   'alpha' : 0.52838575},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 2, rdo factor=2
                'init_7rv_1e_1_2_2' :  {'twist_cp' : np.array([4.82659706, 10, 5]),
                                   'thickness_cp' : 1.e-3 * np.ones(3),
                                   'sweep' : 17.59063958,
                                   'alpha' : -0.09187924},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 3, rdo factor=2
                'init_7rv_1e_1_3_2' :  {'twist_cp' : np.array([4.9103329, 10, 5]),
                                   'thickness_cp' : 1.e-3 * np.ones(3),
                                   'sweep' : 17.46974998,
                                   'alpha' : -0.28339684},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 4, rdo factor=2
                'init_7rv_1e_1_4_2' :  {'twist_cp' : np.array([4.85883309, 10, 5]),
                                   'thickness_cp' : 1.e-3 * np.ones(3),
                                   'sweep' : 17.48184078,
                                   'alpha' : -0.28440502},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 5, rdo factor=2
                'init_7rv_1e_1_5_2' :  {'twist_cp' : np.array([4.87843855, 10, 5]),
                                   'thickness_cp' : 1.e-3 * np.ones(3),
                                   'sweep' : 17.4693496,
                                   'alpha' : -0.3003593},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 6, rdo factor=2
                'init_7rv_1e_1_6_2' :  {'twist_cp' : np.array([4.92920057, 10, 5]),
                                   'thickness_cp' : 1.e-3 * np.ones(3),
                                   'sweep' : 17.45015421,
                                   'alpha' : -0.31792675},
                # 7rv, Full Monte Carlo simulation
                'init_7rv_mc' :  {'twist_cp' : np.array([4.93195234, 10, 5]),
                             'thickness_cp' : 1.e-3 * np.ones(3),
                             'sweep' : 17.48339411,
                             'alpha' : -0.25422074},

                'init_7rv_full_2' : {'twist_cp' : np.array([4.87264491, 10, 5]),
                                'thickness_cp' : 1.e-3 * np.ones(3),
                                'sweep' : 17.54128459,
                                'alpha' : -0.17390615},

                # 'init_7rv_1e_1_2_3' : {'twist_cp' : np.array([4.819646, 10, 5]),
                #                    'thickness_cp' : 1.e-3 * np.ones(3),
                #                    'sweep' : 17.5924635,
                #                    'alpha' : -0.09117171},

                # Seed: deterministic optimal desig, 7RV, sample_radius=1.e-1, n_dominant_dir = 1, rdo factor=2
                'det_7rv_1e_1_1_2' :  {'twist_cp' : np.array([4.58967023, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.93484778,
                                       'alpha' : 0.53324078},

                'det_7rv_1e_1_2_2' :  {'twist_cp' : np.array([4.83860053, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.58985573,
                                       'alpha' : -0.0877428},

                'det_7rv_1e_1_3_2' :  {'twist_cp' : np.array([4.92359245, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.46207284,
                                       'alpha' : -0.29208272},

                'det_7rv_1e_1_4_2' :  {'twist_cp' : np.array([4.87498211, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.49237078,
                                       'alpha' : -0.25990538},

                'det_7rv_1e_1_5_2' :  {'twist_cp' : np.array([4.92460331, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.45721288,
                                       'alpha' : -0.30688852},

                'det_7rv_1e_1_6_2' :  {'twist_cp' : np.array([4.92935159, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.45163998,
                                       'alpha' : -0.31528416},

                # Monte Carlo, init_design, 7rv dominant_dir=2 rdo_factor=2 load factor=1
                'mc_init_7rv_1e_1_1_2' :  {'twist_cp' : np.array([]),
                                           'thickness_cp' : 1.e-3 * np.ones(3),
                                           'sweep' : 0.0,
                                           'alpha' : 0.0},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 2, rdo factor=2
                'mc_init_7rv_1e_1_2_2' :  {'twist_cp' : np.array([4.837423, 10, 5]),
                                           'thickness_cp' : 1.e-3 * np.ones(3),
                                           'sweep' : 17.57359,
                                           'alpha' : -0.11741889},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 3, rdo factor=2
                'mc_init_7rv_1e_1_3_2' :  {'twist_cp' : np.array([4.955151, 10, 5]),
                                           'thickness_cp' : 1.e-3 * np.ones(3),
                                           'sweep' : 17.3953892,
                                           'alpha' : -0.395885},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 4, rdo factor=2
                'mc_init_7rv_1e_1_4_2' :  {'twist_cp' : np.array([4.779176, 10, 5]),
                                           'thickness_cp' : 1.e-3 * np.ones(3),
                                           'sweep' : 17.62265572,
                                           'alpha' : -0.05036577},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 5, rdo factor=2
                'mc_init_7rv_1e_1_5_2' :  {'twist_cp' : np.array([4.796746, 10, 5]),
                                           'thickness_cp' : 1.e-3 * np.ones(3),
                                           'sweep' : 17.61528933,
                                           'alpha' : -0.05819699},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 6, rdo factor=2
                'mc_init_7rv_1e_1_6_2' :  {'twist_cp' : np.array([4.797755, 10, 5]),
                                           'thickness_cp' : 1.e-3 * np.ones(3),
                                           'sweep' : 17.68482033,
                                           'alpha' : 0.07756599},

                'mc_init_7rv_full_2' : {'twist_cp' : np.array([4.931952, 10, 5]),
                                        'thickness_cp' : 1.e-3 * np.ones(3),
                                        'sweep' : 17.48339411,
                                        'alpha' : -0.25422074},

                # Monte Carlo 100 samples, init_design, 7rv, dominant_dir=1, rdo_factor=2, load_factor=1
                'mc100_init_7rv_1e_1_1_2' : {'twist_cp' : np.array([ 4.46700079, 10., 5.]) ,
                                             'thickness_cp' : 1.e-3 * np.ones(3),
                                             'sweep' : 18.06591647,
                                             'alpha' : 0.78298047,
                                            },
                # Monte Carlo 100 samples, init_design, 7rv, dominant_dir=2, rdo_factor=2, load_factor=1
                'mc100_init_7rv_1e_1_2_2' : {'twist_cp' : np.array([4.78863268, 10., 5.]),
                                             'thickness_cp' : 1.e-3 * np.ones(3),
                                             'sweep' : 17.64874499,
                                             'alpha' : 0.00643961,
                                            },
                # Monte Carlo 100 samples, init_design, 7rv, dominant_dir=3, rdo_factor=2, load_factor=1
                'mc100_init_7rv_1e_1_3_2' : {'twist_cp' : np.array([ 4.82232343, 10., 5.]),
                                             'thickness_cp' : 1.e-3 * np.ones(3),
                                             'sweep' : 17.60289199,
                                             'alpha' : -0.0689488,
                                            },
                # Monte Carlo 100 samples, init_design, 7rv, dominant_dir=4, rdo_factor=2, load_factor=1
                'mc100_init_7rv_1e_1_4_2' : {'twist_cp' : np.array([ 4.69792788, 10., 5.]),
                                             'thickness_cp' : 1.e-3 * np.ones(3),
                                             'sweep' : 17.67365726,
                                             'alpha' : 0.00910094,
                                            },
                # Monte Carlo 100 samples, init_design, 7rv, dominant_dir=5, rdo_factor=2, load_factor=1
                'mc100_init_7rv_1e_1_5_2' : {'twist_cp' : np.array([ 4.73816242, 10., 5.]),
                                             'thickness_cp' : 1.e-3 * np.ones(3),
                                             'sweep' : 17.64767038,
                                             'alpha' : -0.02781194,
                                            },
                # Monte Carlo 100 samples, init_design, 7rv, dominant_dir=6, rdo_factor=2, load_factor=1
                'mc100_init_7rv_1e_1_6_2' : {'twist_cp' : np.array([4.68899128, 10., 5.]),
                                             'thickness_cp' : np.array([0.001, 0.001, 0.00108472]),
                                             'sweep' : 17.61094628,
                                             'alpha' : -0.16934135,
                                            },

                # Active subspace, init design. 7rv, dominant dir= 2, rdo factor=2.0, load factor = 1
                'act_init_7rv_1_2_lf1' : {'twist_cp' : np.array([4.18875197, 10, 5]),
                                          'thickness_cp' : np.array([0.001, 0.001, 0.00121679]),
                                          'sweep' :18.11711896,
                                          'alpha' : 0.69372124},

                'act_init_7rv_2_2_lf1' : {'twist_cp' : np.array([4.81509051, 10, 5]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' :17.6069306,
                                          'alpha' : -0.06437804},

                'act_init_7rv_3_2_lf1' : {'twist_cp' : np.array([4.8662098, 10, 5]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' :17.5308038,
                                          'alpha' : -0.18833107},

                'act_init_7rv_4_2_lf1' : {'twist_cp' : np.array([4.85453896, 10, 5]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' :17.55328352,
                                          'alpha' : -0.15536896},

                'act_init_7rv_5_2_lf1' : {'twist_cp' : np.array([4.87966718, 10, 5]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' :17.52586469,
                                          'alpha' : -0.19962768},

                'act_init_7rv_6_2_lf1' : {'twist_cp' : np.array([4.88126341, 10, 5]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' : 17.52524598,
                                          'alpha' : -0.20026676},

                # Seed: init design, 7RV, sample_radius=1.e-1, n_dominant_dir = 1, rdo factor=2 load factor = 2.5
                'init_7rv_1e_1_1_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                              'thickness_cp' : np.array([0.001, 0.001, 0.00232218]),
                                              'sweep' : 18.90483727,
                                              'alpha' : 10.},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 2, rdo factor=2
                'init_7rv_1e_1_2_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                              'thickness_cp' : np.array([0.001, 0.001, 0.00234215]),
                                              'sweep' : 18.90303949,
                                              'alpha' : 10.},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 3, rdo factor=2
                'init_7rv_1e_1_3_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                              'thickness_cp' : np.array([0.001, 0.001, 0.00235118]),
                                              'sweep' : 18.90210583,
                                              'alpha' : 10.},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 4, rdo factor=2
                'init_7rv_1e_1_4_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                              'thickness_cp' : np.array([0.001, 0.001, 0.00235733]),
                                              'sweep' : 18.89959251,
                                              'alpha' : 10.},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 5, rdo factor=2
                'init_7rv_1e_1_5_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                              'thickness_cp' : np.array([0.001, 0.001, 0.0023529]),
                                              'sweep' : 18.90014016,
                                              'alpha' : 10.},
                # 7 RV, sample radius = 1.e-1, n_dominant_dir = 6, rdo factor=2
                'init_7rv_1e_1_6_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                              'thickness_cp' : np.array([0.001, 0.001, 0.00234452]),
                                              'sweep' : 18.90120998,
                                              'alpha' : 10.},

                # Seed: deterministic optimal desig, 7RV, sample_radius=1.e-1, n_dominant_dir = 1, rdo factor=2 load factor = 2.5
                'det_7rv_1e_1_1_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                             'thickness_cp' : np.array([0.001, 0.001, 0.00232253]),
                                             'sweep' : 18.9047977,
                                             'alpha' : 10.},

                'det_7rv_1e_1_2_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                             'thickness_cp' : np.array([0.001, 0.001, 0.00234304]),
                                             'sweep' : 18.90294882,
                                             'alpha' : 10.},

                'det_7rv_1e_1_3_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                             'thickness_cp' : np.array([0.001, 0.001, 0.00235205]),
                                             'sweep' : 18.90200067,
                                             'alpha' : 10.},

                'det_7rv_1e_1_4_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                             'thickness_cp' : np.array([0.001, 0.001, 0.00235167]),
                                             'sweep' : 18.90063552,
                                             'alpha' : 10.},

                'det_7rv_1e_1_5_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                             'thickness_cp' : np.array([0.001, 0.001, 0.00234707]),
                                             'sweep' : 18.90112309,
                                             'alpha' : 10.},

                'det_7rv_1e_1_6_2_lf2.5' :  {'twist_cp' : np.array([10., 10., 5.]),
                                             'thickness_cp' : np.array([0.001, 0.001, 0.00234432]),
                                             'sweep' : 18.90126052,
                                             'alpha' : 10.},
                }
