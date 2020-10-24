"""
Module with event processing utilities, main functionalities:
* transform data from event format (row per event) into real-time-prediction 
    format as in the thesis, i.e. as in Figure 4.3, but with a large degree of
    customizability. "events_to_rt_pred" is the most high-level function that
    offers this functionality.
* performing "templating" as described in the thesis with a self-specified
    variable serving as the template.
* missing value imputation based on the time-between-events for each variable,
    with customizable aggregation functions over the time between events
    over the variables.
* extract the time between events for different types of events from
    an event log data format.
* performance-crucial parts are written purely with Pandas vectorized 
    functions and thus fast over large dataframes. Developed on an event log
    with 4M+ events, conversion runs in minutes.

Why to use this module?
* To easily re-use the functionality from above in other experiments that
    involve different events, different variables and different template
    variables.

When not to use this module:
* If Pandas supports forward filling (.ffill function) with a maximum amount of
    time to fill forward in a future version.
    The version as of this writing (1.0.3) only supports forward
    filling with a maximum number of fills, not with a maximum amount of time.
    If such functionality is added to Pandas in the future, then this module
    will probably become redundant.
"""

import pandas as pd

def get_time_between_events(df, func):
    """
    Get a series with the selected statistic of the time between events for 
        each variable.
    
    inputs
    -----
    * df: input dataframe with a unique index and columns:
        case_id (int) | timestamps (pd datetime) | variable (str)
    * func: function to use for aggregating the time between events of each 
        variable. See pandas.DataFrame.agg for options. The function is 
        omputed for each variable individually.
    """
    cid, ts, var = df.columns  # store colnames for easy reference
    dfi = df.sort_values([cid, ts]) #Internal, sorted df
    
    #Retrieve times-between-events for the variables, taking into account cases.
    #As seconds, otherwise aggregation functions do not work
    df_diff = dfi.groupby([cid, var])[ts]\
        .diff()\
        .dt.total_seconds()\
        .to_frame('diff')
    df_diff = df_diff.merge(df, left_index=True, right_index=True)[[var, 'diff']]

    #Convert back to timedelta, dependent on whether one or more output columns
    if (type(func) != list):
        out = pd.to_timedelta(
            df_diff.groupby(var)['diff'].agg(func), unit='s', errors='coerce'
        ) #One column
    else:
        out = df_diff.groupby(var)['diff'].agg(func) #multiple columns
        for col in out.columns:
            out[col] = pd.to_timedelta(out[col].values, unit='s', errors='coerce')
    
    return out

def time_since_last(df):
    """
    Find the time since the last entry (TSL) in a dataframe with the events 
        OF A SINGLE VARIABLE.
    
    inputs
    -----
    * df: dataframe with 3 columns in the following order:
        case_id (int) | timestamps (pd datetime) | entries (float or int)
    """
    #Setup
    dfi = df.copy(deep=True) #Internal copy for processing
    cid, ts, ent = df.columns #Store colnames for easy reference
    dfi.sort_values([cid, ts], inplace=True) #Sort by case & time - otherwise computation is incorrect
    
    #Generate time difference column
    dfi['td'] = dfi\
        .groupby(cid)\
        [ts]\
        .diff()
    dfi['td'] = dfi['td'].dt.seconds #To second for making cumsum possible
    
    #Manually set the first entries for each case to 0 to avoid issues with forward filling from the first entry
    first_indices = dfi[[cid, ts, ent]]\
        .reset_index(drop=False)\
        .groupby(cid)['index']\
        .first()\
        .values
    dfi.loc[first_indices, 'td'] = 0
    
    #Cumsum, shifting & filling
    dfi['td_cumsum'] = dfi.groupby(cid)['td'].cumsum()
    dfi['td_cumsum_mod'] = dfi['td_cumsum'] * dfi[ent] / dfi[ent] #Add NaNs in places where there was no entry
    dfi['td_cumsum_mod'] = dfi.groupby(cid)['td_cumsum_mod'].shift(1) #Shift 1
    dfi['td_cumsum_mod'] = dfi.groupby(cid)['td_cumsum_mod'].ffill() #ffil
    dfi['TSL'] = dfi['td_cumsum'] - dfi['td_cumsum_mod'] #Get time since the last measurement (TSL)
    dfi.loc[~dfi[ent].isnull().values, 'TSL'] = 0 #Correct the previous - add 0 in all places with entries
    
    #Dtype conversion and output formatting
    dfi['TSL'] = pd.to_timedelta(dfi['TSL'], unit='seconds') #Convert to timedelta format
    out = dfi.loc[df.index, 'TSL']
    
    return out

def ffill_with_timelimit(df, time_limit):
    """
    Forward-fill events recorded over time per-case with a time limit for carrying values forward
        in a dataframe with the events OF A SINGLE VARIABLE.
    
    inputs
    -----
    * df: dataframe with 3 columns in the following order:
        case_id (int) | timestamps (pd datetime) | entries (float or int)
    * time_limit: the maximum amount of time that the entries are allowed to be carried forward.
        timedelta or string indicating a timedelta (see pandas.to_timedelta)
    """    
    #--Setup
    #Convert time limit to correct format
    if (type(time_limit) == str):
        tl = pd.to_timedelta(time_limit)
    else:
        tl = time_limit
    cid, ts, ent = df.columns #Store colnames for easy reference
    
    #Computation
    dfi = time_since_last(df).to_frame() #Get time since last measurement as a df for internal usage
    dfi['ffil_ok'] = dfi['TSL'] <= tl #Mask whether we should ffil
    dfi['ful_ffil'] = df.groupby(cid)[ent].ffill()
    dfi.loc[dfi['ffil_ok'], 'out'] = dfi.loc[dfi['ffil_ok'], 'ful_ffil'] 
    
    return dfi['out']

def events_to_rt_pred(df, max_cft, template_var=None, output_max_cfts=False, verbose=False):
    """
    Convert a dataframe with different types of events (or measurements) over time, 
        and that belong to a case, into a format that enables real-time predictions. I.e.:
        -------------------------------------------------------------------------------------
        in: case_id (int)  | timestamps (pd datetime) | variable (str)       | value (float/int)
        out: case_id (int) | timestamps (pd datetime) | var1_val (float/int) | var2_value |....
        -------------------------------------------------------------------------------------
        
    Address sparsity with forward filling using a self-chosen variant of the time between events
        as the maximum carry-forward time (max_cft). This takes into account the cases,
        i.e. a time to event must be within the same case.
    
    See the following Jupyter notebooks for examples on the performed conversion of this function:
    * 02) Filling Strategies Tests.ipynb
    * 03) Observation vs Case-Based Splitting Tests.ipynb
    * 04) Events to Modeling Format.ipynb
        
    inputs
    -----
    * df: input dataframe with a unique index and columns:
        case_id (int) | timestamps (pd datetime) | variable (str) | value (float or int)
    
    * max_cft: maximum carry forward time, i.e. how long we are allowed to carry the value 
        of a measurement forward in time. 3 options:
        * function to use for aggregating the data. See pandas.DataFrame.agg
            for options. The function is computed for each variable individually.
        * dictionary mapping each variable to static self-chosen max_cf, i.e.
            a timedelta or string of time. See pandas.to_timedelta for possible 
            values for string of time.
        * Scalar timedelta, meaning this max_cf will be used for
            all features.
    
    * template_var: variable that serves as the template (i.e. only keep rows with that event).
        default = None, implying no template (keep all rows).
    
    * output_max_cf: whether the used time between events should be returned. Default=False.
    
    * verbose: whether to print output of the progress of the calculation
    
    """
    #Check whether the index is unique
    if (not df.index.is_unique):
        raise Exception("Index was not unique. Please create a unique index (e.g. with .reset_index())")
        
    cid, ts, var, val = df.columns #store colnames for easy reference
    
    #Pivot the df & sort on case_id, timestamp
    dfi = df.pivot(index=None, columns=var, values=val) #Pivot the df
    dfi = df[[cid, ts]].merge(dfi, left_index=True, right_index=True) #Merge back cid & ts
    dfi.sort_values([cid, ts], inplace=True) #Sort for later processing
    
    #Compute the TTEs if necessary
    if (type(max_cft) == str):
        max_cfts = get_time_between_events(df[[cid, ts, var]], max_cft)
    
    #Perform the forward filling of each column based on max_cft
    for i, variable in enumerate(dfi.columns[2:]):
        if (verbose):
            print('Filling variable {}: {}'.format(i,variable))
        if (type(max_cft)==pd.Timedelta):
            dfi[variable] = ffill_with_timelimit(dfi[[cid, ts, variable]], max_cft)
        else:
            dfi[variable] = ffill_with_timelimit(dfi[[cid, ts, variable]], max_cft[variable])
    
    #Merge duplicate timestamps by averaging
    #may not be appropriate in other contexts than ours (e.g. with event counts)
    df_out = dfi.groupby([cid, ts]).mean().reset_index(drop=False)
    
    #Filter rows on only those of the template of required
    if (template_var != None):
        #Find the case_ids & timestamps corresponding to the template variable
        templ_cid_ts = df[df[var] == template_var][[cid, ts]].drop_duplicates().values
        indices = [tuple(cid_ts) for cid_ts in templ_cid_ts] #Need the indices as tuples to use .loc
        
        #Filter the output df on the timestamps in the template
        df_out = df_out.set_index([cid, ts])\
            .sort_index()\
            .loc[indices]\
            .reset_index(drop=False)

    if (output_max_cfts):
        return df_out, max_cfts
    else:
        return df_out
