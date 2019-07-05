# -*- coding: utf-8 -*-
"""
Created by Liu Zongying on June 2019

"""

import mdp
import Oger
import io_simple_xp
import error_measure
import plot_tools
import diff_measure
import weight_function


def compute_error(error_obj, _states_out, _curr_teachers, _subset, verbose=False):
    all_meaning_err = []
    all_sentence_err = []
    for i in range(len(_curr_teachers)):
        aa = _states_out[i]
        bb = _curr_teachers[i]
        full_info_err = error_obj.compute_error(input_signal=_states_out[i],
                                                target_signal=_curr_teachers[i],
                                                verbose=verbose)
        (err_avg_on_signals, global_err_answer, nr_of_erroneous_SW, total_nr_of_pertinent_SW, NVa_correct,
         NVa_erroneous) = full_info_err
        all_meaning_err.append(err_avg_on_signals)
        all_sentence_err.append(global_err_answer)
    mean_me = mdp.numx.mean(all_meaning_err)
    std_me = mdp.numx.std(all_meaning_err)
    median_me = mdp.numx.median(all_meaning_err)
    mean_se = mdp.numx.mean(all_sentence_err)
    std_se = mdp.numx.std(all_sentence_err)
    median_se = mdp.numx.median(all_sentence_err)
    return (mean_me, std_me, median_me, mean_se, std_se, median_se)


def simulation(root_file_name=None, N=100, sr=1, tau=6, act_time=20, subset=range(15, 41),
               in_scal=0.75, ridge=10 ** -9, n_folds=10, seed=None,
               comp_err=True, plot_output=True, comp_diff_states=True, verbose=False):
    """
    Inputs:
        N: number of internal units
        sr: spectral radius (scale factor for the internal weight matrix)
        tau: time constant
        act_time: activation time (corresponds to 1/delta(t) in internal units equation)
        subset: list containing the indices of the grammatical constructions that will be used
        in_scal: input scaling (scale factor for the input weight matrix)
        ridge: parameter of the ridge regression
        n_folds: select the type of learning:
            0: train and test on all data
            -1: leave one out
            x: cross validation with x folds (x nonnegative)
        seed: seed for the random number generator. If None, it will be initialized on computer time clock
        comp_err: if True, computes the meaning and sentence errors
        plot_output: if True, plots the readout (output) units activity
                    if 'comp_diff_states' is True also, plots the sum of instantaneous change in the readout activity
        comp_diff_states: if True, computes the sum of instantaneous change in the readout activity
        verbose: if True, display more information

    Output:
        (mean of meaning errors, standard deviation of meaning errors,
            mean of sentence errors, standard deviation of sentence errors)
    """

    #    keep_internal_states = True # to use, uncomment all lines with keep_internal_states
    verbose_each_step = True

    d_io = {}
    d_io['subset'] = subset
    d_io['act_time'] = act_time  # equal to 1/delta_t
    # Add supplementary time steps after input stimulus is finished
    #        Note that the error is done on the last time step when executing the reservoir;
    #        If d_io['suppl_pause_at_the_end'] is changed, so will change the error measure.
    d_io['suppl_pause_at_the_end'] = 0  # 1
    d_io['initial_pause'] = True
    (d_io['l_input'], d_io['l_data']) = io_simple_xp.get_corpus()
    (d_io['l_output'], d_io['l_teacher']) = io_simple_xp.get_coded_meaning()

    ## Generate the input stimulus
    inputs = io_simple_xp.generate_stim_input(d_io=d_io, verbose=verbose)
    ## Generate the teacher output
    teacher_outputs = io_simple_xp.generate_teacher_output(d_io=d_io, verbose=verbose)
    d_io['dim_input'] = inputs[0].shape[1]
    if verbose:
        print "len(inputs)", len(inputs)
        print "inputs[0].shape", inputs[0].shape
        print "len(teacher_outputs)", len(teacher_outputs)
        print "d_io['subset']", d_io['subset']

    ############## original input weight ########################################
    # w_R = mat_gen.generate_internal_weights(N=N, spectral_radius=sr, proba=0.1,
    #                                         seed=seed, verbose=verbose)
    # w_in_R = mat_gen.generate_input_weights(nbr_neuron=N, dim_input=d_io['dim_input'],
    #                                         input_scaling=in_scal, proba=0.1, verbose=verbose)

    ############ initialization input weight by Xavier ###########################
    w_R = weight_function.wfunction(output_dim=N, input_scaling=in_scal, spectral_radius=sr, prob=0.1)
    w_in_R = weight_function.w_inputweight(_input_dim=d_io['dim_input'], _output_dim=N, input_scaling=in_scal)
    ############# initialization input weight by method3 ##########################
    # [w_in_R, w_R] = weight_function_paper.initiliaion_function(ni=13, nr=100, rho=0.7, scaling=0.10, connectivity=0.1)
    ## Instanciate reservoir, read-out and flow
    reservoir = Oger.nodes.LeakyReservoirNode(nonlin_func=mdp.numx.tanh, w=w_R, w_in=w_in_R,
                                              input_dim=d_io['dim_input'], output_dim=N,
                                              leak_rate=(1. / (tau * d_io['act_time'])))
    read_out = Oger.nodes.RidgeRegressionNode(ridge_param=ridge, use_pinv=True, with_bias=True)
    flow = mdp.Flow([reservoir, read_out])

    ## Instanciate error parameters
    err_obj = error_measure.thematic_role_error(d_io=d_io)

    #    if keep_internal_states:
    #        Oger.utils.make_inspectable(mdp.Flow)

    ## Determining the training and testing sets:
    # train = test
    if n_folds is None or n_folds == 0:
        #        train_indices = [mdp.numx.array(d_io['subset'])]
        train_indices = [range(len(d_io['subset']))]
        test_indices = train_indices
    # leave-one-out
    elif n_folds == -1:
        train_indices, test_indices = Oger.evaluation.leave_one_out(len(d_io['subset']))
    # cross validation
    else:
        train_indices, test_indices = Oger.evaluation.n_fold_random(
            n_samples=len(d_io['subset']),
            n_folds=n_folds)

    # Initialize variables to keep performance mesures over all test sets:
    all_mean_meaning_err = mdp.numx.zeros(len(train_indices))
    all_mean_sent_err = mdp.numx.zeros(len(train_indices))

    for i in range(len(train_indices)):
        f_copy = flow.copy()
        #        if keep_internal_states:
        #            # works for one loop only (i.e. n_folds=0)
        #            f_copy = flow
        curr_inputs_train = [inputs[x] for x in train_indices[i]]
        curr_teachers_train = [teacher_outputs[x] for x in train_indices[i]]
        curr_inputs_test = [inputs[x] for x in test_indices[i]]
        curr_teachers_test = [teacher_outputs[x] for x in test_indices[i]]

        ## Learning
        learning_data = [curr_inputs_train, zip(curr_inputs_train, curr_teachers_train)]

        f_copy.train(learning_data)
        ## Testing
        states_out_test = len(curr_inputs_test) * [None]

        #        if keep_internal_states:
        #            internal_states_test = len(curr_inputs_test)*[None]
        for idx_out in range(len(curr_inputs_test)):
            states_out_test[idx_out] = f_copy(curr_inputs_test[idx_out])
        #            if keep_internal_states==True:
        #                #saving internal states of LeakyReservoirNode neurons
        #                internal_states_test[idx_out] = f_copy.inspect(reservoir)

        subset = [d_io['subset'][x] for x in test_indices[i]]
        subtitle = "test_data " + str(subset[0])
        if verbose_each_step:
            print "****************************************************"
            print "subset_test: test_indices[i]:", test_indices[i]
            print "subset: d_io['subset'][test_indices[i]:", subset

        if comp_err:
            ## compute home-made error version 2
            result_comp_err = compute_error(error_obj=err_obj, _states_out=states_out_test,
                                            _curr_teachers=curr_teachers_test, _subset=subset,
                                            verbose=False)
            (mean_meaning_err, std_meaning_err, median_meaning_err, mean_sent_err, std_sent_err,
             median_sent_err) = result_comp_err
        else:
            (mean_meaning_err, std_meaning_err, median_meaning_err, mean_sent_err, std_sent_err, median_sent_err) = (
            -1, -1, -1, -1, -1, -1)

        if verbose_each_step:
            print "mean meaning_err: ", mean_meaning_err
            print "std meaning_err:", std_meaning_err
            print "median meaning_err:", median_meaning_err
            print "mean sent_err", mean_sent_err
            print "std sent_err", std_sent_err
            print "median sent_err", median_sent_err
            print "****************************************************"

        ## Compute derivative-like of readout activity
        if comp_diff_states:
            diff = diff_measure.amount_of_change(states_out=states_out_test)
            s_diff = diff_measure.sum_amount_of_change(diff=diff, return_as_tuple=False)

        ## Optional Plots
        if plot_output:
            plot_tools.plot_output(forced_subset=subset, _outputs=states_out_test, d_io=d_io,
                                   save_pdf=True, nr_nouns=4, nr_verbs=2, root_file_name=root_file_name,
                                   subtitle=subtitle + "__m-err" + str(mean_meaning_err) + "_s-err" + str(
                                       mean_sent_err), window=0,
                                   verbose=False, y_lim=[-1.5, 1.5])
            if comp_diff_states:
                plot_tools.plot_with_output_fashion(l_array=s_diff, subset=subset, d_io=d_io,
                                                    root_file_name=root_file_name, subtitle=subtitle + "_sum_diff",
                                                    legend=["sum_diff", "abs_sum_diff", "abs_max_diff"],
                                                    y_lim=None, verbose=False)
        #            if keep_internal_states:
        #                plot_tools.plot_array_in_file(root_file_name=root_file_name, array_=internal_states_test,
        #                                   data_subset=None, titles_subset=None, plot_slice=None,
        #                                   title="_internal_states", subtitle="", legend_=None)

        ## save infos in lists
        all_mean_meaning_err[i] = mean_meaning_err
        all_mean_sent_err[i] = mean_sent_err

    if verbose:
        print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*"
        print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*"
        print "Mean error for all mean_meaning_err:", str(mdp.numx.mean(all_mean_meaning_err))
        print "Std for all mean_meaning_err:", str(mdp.numx.std(all_mean_meaning_err))
        print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*"
        print "Mean error for all mean_sent_err:", str(mdp.numx.mean(all_mean_sent_err))
        print "Std for all mean_sent_err:", str(mdp.numx.std(all_mean_sent_err))
        print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*"
    return (mdp.numx.mean(all_mean_meaning_err), mdp.numx.std(all_mean_meaning_err), mdp.numx.mean(all_mean_sent_err),
            mdp.numx.std(all_mean_sent_err))


def multi_sim(nr_instance=1, seed=None, *args, **kwargs):
    list_mean_merr = []
    list_mean_serr = []
    if seed is not None:
        curr_time = seed  # 0
        print "seed defined (not random): " + str(curr_time)
    else:
        import time
        #        curr_time = int(time.time()*10**6)
        curr_time = int((time.time() * 10 ** 6) % int(time.time()))
        print "random seed generated: " + str(curr_time)
    for i in range(nr_instance):
        (mean_merr, std_me, mean_serr, std_se) = simulation(seed=i + curr_time, *args, **kwargs)
        list_mean_merr.append(mean_merr)
        list_mean_serr.append(mean_serr)
    return (mdp.numx.mean(list_mean_merr), mdp.numx.std(list_mean_merr), mdp.numx.mean(list_mean_serr),
            mdp.numx.std(list_mean_serr))


if __name__ == '__main__':
    ######################################################################""
    # Defining parameters for the input and corpus
    # change "subset" in order to change the data set used
    ######################################################################""

    subset = range(15,41) # Exp1, Exp 3
    # subset = range(0, 41)

    #    (mean_me, std_me, mean_se, std_se) = simulation(
    #        root_file_name='../../RES_TEMP/run_simple_xp',
    #        N=100, sr=1, tau=6, act_time=20, subset=subset,
    #        in_scal=0.75, ridge=10**-9, n_folds=-1, seed=1,
    #        comp_err=True, plot_output=True, comp_diff_states=True, verbose=False)
    (mean_me, std_me, mean_se, std_se) = multi_sim(nr_instance=1,
                                                   root_file_name='../RES_TEMP/IP-ESN',
                                                   N=100, sr=1, tau=6, act_time=20, subset=subset,
                                                   in_scal=0.75, ridge=10 ** -9, n_folds=-1, seed=1,
                                                   comp_err=True, plot_output=False, comp_diff_states=True,
                                                   verbose=True)
    print (mean_me, std_me, mean_se, std_se)

