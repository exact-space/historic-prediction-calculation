import warnings 
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from logzero import logger
import joblib
logger.info("joblib version: "+joblib.__version__)
import pandas as pd
import numpy as np
import os
import platform
import time
version = platform.python_version().split(".")[0]
if version == "3":
  import app_config.app_config as cfg
  import timeseries.timeseries as ts
elif version == "2":
  import app_config as cfg
  import timeseries as ts
import datetime
config = cfg.getconfig()
qr = ts.timeseriesquery()
meta = ts.timeseriesmeta()
import requests
import json
modelBias=0
sensitivity=1
pro = ts.timeseriesprocessors()
qr = ts.timeseriesquery()
meta = ts.timeseriesmeta()



def getData1(taglist,timeType,qr, key = None,unitId = None, aggregators = [{"name":"avg","sampling_value":1,"sampling_unit":"minutes"}]):

    qr.addMetrics(taglist)
    if(timeType["type"]=="date"):
        qr.chooseTimeType("date",{"start_absolute":timeType["start"], "end_absolute":timeType["end"]})

    elif(timeType["type"]=="relative"):
        qr.chooseTimeType("relative",{"start_unit":timeType["start"], "start_value":timeType["end"]})


    elif(timeType["type"]=="absolute"):
        qr.chooseTimeType("absolute",{"start_absolute":timeType["start"], "end_absolute":timeType["end"]})


    else:
        logger.info('Error')
        logger.info('Improper timetype[type]')

    if aggregators != None:
        qr.addAggregators(aggregators)

    if ((key) and (key == "simulation")):
        qr.submitQuery("simulation",unitId)
    else:
        key = None
        qr.submitQuery(key,unitId)


    qr.formatResultAsDF()
    try:

        df = qr.resultset["results"][0]["data"]
        return df
    except Exception as e:
        logger.info('Data Not Found getData1 '+e)
        return pd.DataFrame()


def download_model(modelconf):
    filename = modelconf["id"]+"_"+modelconf["deployVersion"]+".pkl"
    url = config["api"]["meta"]+'/attachments/models/download/'+filename
    logger.info("file url: "+url)
    r = requests.get(url)

    if r.status_code == 404:
        filename = modelconf["id"]+"_"+modelconf["deployVersion"]+".h5"
        url = config["api"]["meta"]+'/attachments/models/download/'+filename
        logger.info(url)
        r = requests.get(url)
        logger.info("response h5.................. "+r.status_code)
        with open("./"+filename, 'wb',buffering=1) as f:
            f.write(r.content)
        return filename
    else:
        with open("./"+filename, 'wb',buffering=1) as f:
            f.write(r.content)
        return filename



def pred(modelconfig, unitId, modelId, tags, timeperiod):
    #create timeType
    try:
        df= getData1(tags,timeperiod,qr)
        df.set_index('time', inplace=True)
        #logger.info(len(df))
        for column in df.columns:
            df[column].fillna(method='bfill', limit=5, inplace=True)
        df.dropna(inplace=True, axis=0)
        modelconfig['RunHistoricProgress'].append(f'Received data for selected time period. {df.shape}')
        meta.updateModel(modelconfig,unitId)
    except Exception as e:
        logger.info('Not able to get or process data from kairos')
        logger.info(e)
        modelconfig['RunHistoricProgress'].append('Not able to get data from kairos')
        meta.updateModel(modelconfig,unitId)
        exit()
    #logger.info(len(df))
    #df = df.head(50)
    logger.info(f'df@@@@@@@@@ {df}')
    #exit()
    errdist={}
    output_tag=[]
    for i in modelconfig['performance']:    
        if i.get("modelVersion")==modelconfig['deployVersion']:
            if "errDist" in list(i.keys()):
                output_tag.append(i["outputTag"])
                selectedVars = i['selectedVars']
                errdist=i["errDist"]
            else:
                modelconfig['RunHistoricProgress'].append('Check the err dist of deployed model.')
                meta.updateModel(modelconfig,unitId)
                exit()

    modelconfig['RunHistoricProgress'].append('Calculating values')
    meta.updateModel(modelconfig,unitId)
    try:
        model_name = download_model(modelconfig)
    except Exception as err:
        logger.info("Failed to download model for "+str(model['id']))
        modelconfig['RunHistoricProgress'].append('Not able to download model file')
        meta.updateModel(modelconfig,unitId)
        exit()
    
    #model_name = modelconfig['id'] + "_" + modelconfig['deployVersion']+".pkl"
    try:
        X_test_val = []
        logger.info("                                                                        ")
        try:
            logger.info(model_name)
            grid = joblib.load(model_name)
        except:
            try:
                grid = load_model(model_name)
            except:
                logger.info('cannot load model with joblib or load_model')
                modelconfig['RunHistoricProgress'].append('Not able to load model file')
                meta.updateModel(modelconfig,unitId)
                exit()
        result_dict = {}
        actual_value={}

        for index, row in df.iterrows():
            values_dict = {col: row[col] for col in selectedVars}
            result_dict[index] = list(values_dict.values())
            values_dict2 = {col: row[col] for col in df.columns if col == output_tag[0]}
            actual_value[index] = list(values_dict2.values())
        #logger.info(result_dict)
        #logger.info(actual_value)
        #exit()
        pred_time=[]
        for key, value in result_dict.items():
            pred_time.append(key)
            X_test_val.append(value)

        #logger.info(pred_time[:10])
        #logger.info(X_test_val[:10])
        #actual_value = df[output_tag[0]].tolist()
        #logger.info(f'@actual {list(actual_value.values())}')
        try:
            y_pred_pipe = grid.predict(X_test_val)
            logger.info(f"prediction@@@@@ {y_pred_pipe}")
            logger.info(len(y_pred_pipe))
        except ValueError:
            logger.info("Input tag data missing. It needs {} values, But it has {} values".format(len(selectedVars),len(X_test_val[0])))
            modelconfig['RunHistoricProgress'].append('Input tag data is missing')
            meta.updateModel(modelconfig,unitId)
            exit()
        #exit()

        logger.info(errdist)
        bucket_list=[]   
        bucketList= [key for key in errdist.keys() if key != "bucketSize"]
        logger.info(f"bucketlist @@@@@ {bucketList} %%%%%% {output_tag[0]}")
        
        bucketListDict={}
        for key in bucketList:
            bucketListDict[round(float(key),3)]=key
        sortedDict=sorted(bucketListDict.keys())
        logger.info(f'bucket list dict {bucketListDict}')
        logger.info(f'sorted dict {sortedDict}')
        for pred in y_pred_pipe:
            if pred != 0:
                for i in range(len(sortedDict)-1):
                    if (pred >= sortedDict[i] and pred< sortedDict[i+1]):
                        bk= bucketListDict[sortedDict[i]]
                        #logger.info(f"matchingBucket {bk}")
                        break
                    elif pred <=sortedDict[0]:
                        bk=bucketListDict[sortedDict[0]]
                        #logger.info(f"matchingBucket {bk}")
                        break
                    elif pred >= sortedDict[-1]:
                        bk= bucketListDict[sortedDict[-1]]
                        #logger.info(f"matchingBucket {bk}")
                        break
                if errdist[bk]["status"]!="invalid":
                    sdNew = errdist[bk]["sdNew"]
                    median=errdist[bk]["median"]
                    bucket_list.append(bk)
                    #logger.info(f"median########### {median}")
                else:
                    bk='invalid'
                    median=0
                    sdNew = "invalid"
                    bucket_list.append(None)
                    #logger.info("invalid bucket")
            else:
                median = 0
                bk = 'invalid'
                sdNew = "invalid"
                bucket_list.append(None)
                #logger.info("errdist not found")
        #logger.info(f'{bucket_list} {len(bucket_list)}')
        c = 0
        while c < len(y_pred_pipe):
            try:
                if bucket_list[c] == None:
                    result_dict.pop(pred_time[c], None)
                else:
                    result_dict[pred_time[c]] = [actual_value[pred_time[c]][0], y_pred_pipe[c], bucket_list[c]]
                c+=1
            except:
                result_dict.pop(pred_time[c], None)
                c+=1
        modelconfig['RunHistoricProgress'].append(f'Prediction Calculation Done')
        meta.updateModel(modelconfig,unitId)
        #logger.info(result_dict)
    except Exception as err:
        logger.info('_______ '+err)

    try:
        data_dict = flags(result_dict, errdist)
        modelconfig['RunHistoricProgress'].append(f'flagModel & pred limits Calculation Done')
        meta.updateModel(modelconfig,unitId)
        return data_dict, output_tag[0]
    except Exception as E:
        logger.info('Not able to calculate flags & limits')
        logger.info(E)
        modelconfig['RunHistoricProgress'].append(f'Something went wrong in flag or limit calculation {E}')
        meta.updateModel(modelconfig,unitId)
        exit()

def flags(dct, err_dist):
    modelFlag = 0
    for key, value in dct.items():
        try:
            prediction = value[1]
            #logger.info(f"********** {prediction}")
            bk = value[2]
            sdNew = err_dist[bk]['sdNew']
            median = err_dist[bk]['median']      
            IQR = err_dist[bk]['q75'] -  err_dist[bk]['q25']
            
            BW = err_dist[bk]['q995'] - err_dist[bk]['q005']
            if sdNew < BW/4:
                sdNew = BW/4
            upperValue1 = round(err_dist[bk]['q995'] + (sdNew*sensitivity),3) + modelBias 
            lowerValue1 = round(err_dist[bk]['q005'] - (sdNew*sensitivity),3) + modelBias
            upperValue2 = upperValue1 + 2*sdNew*sensitivity
            lowerValue2 = lowerValue1 - 2*sdNew*sensitivity
            upperValue3 = upperValue1 + 3*sdNew*sensitivity
            lowerValue3 = lowerValue1 - 3*sdNew*sensitivity
            

            predLw = round(prediction-(median-lowerValue1),3)
            predUp = round(prediction+(upperValue1-median),3)                   
            predLw2 = round(prediction-(median-lowerValue2),3)
            predUp2 = round(prediction+(upperValue2-median),3)
            predLw3 = round(prediction-(median-lowerValue3),3)
            predUp3 = round(prediction+(upperValue3-median),3)
            #logger.info(f"predLw: {predLw} predUp: {predUp}")
            if ((value[0] >= predLw) and (value[0] <= predUp)):
                modelFlag = 0
        
            elif value[0] > predUp and value[0] <= predUp2:
                modelFlag = 1

            elif value[0] > predUp2 and value[0] <= predUp3:
                modelFlag = 2
            
            elif value[0] > predUp3:
                modelFlag = 3
            
            elif value[0] < predLw and value[0] >= predLw2:
                modelFlag = -1

            elif value[0] < predLw2 and value[0] >= predLw3:
                modelFlag = -2
            
            elif value[0] < predLw3:
                modelFlag = -3    
            
            else:
                logger.info("No condition satisfied.")
                
            #logger.info(f"flag######## {modelFlag}")
            dct[key].extend([modelFlag, predLw, predUp])
        except:
            dct.pop(key, None)

    for keys, values in dct.items():
        values.pop(2)
        dct[keys]=values
    #logger.info(dct)
    return dct


def postDataApi(tag,store_vals_to_post):

    url = str(config["api"]["datapoints"])
    #print(url)
    #url="https://data.exactspace.co/kairosapi/api/v1/datapoints/query"
    batch_size = 20000
    for i in range(0, len(store_vals_to_post), batch_size):
        batch = store_vals_to_post[i:i+batch_size]
        body = [{
            "name": str(tag),
            "datapoints": batch,
            "tags":{"type":"raw"}}]
        res = requests.post(url = url,json = body,stream=True)
        time.sleep(2)
        if res.status_code != 204:
            logger.info("Error posting data to KairosDB:"+str(res.content))
        else:
            logger.info("Data posted successfully to KairosDB")
    return res.status_code 

def calculation(modelId = None, startTime=None, endTime=None):
    try:
        url = config['api']['meta']+'/modelpipelines/'+str(modelId)
        reponse = requests.get(url)
        model_json = json.loads(reponse.content)
        unitId = model_json['unitId']
        model_json['RunHistoricProgress'] = []
        if not model_json['deploy']:
            logger.info('Model not deployed.')
            model_json['RunHistoricProgress'].append('Model not deployed')
            meta.updateModel(model_json,unitId)
            exit()
        logger.info("Deployed Version "+str(model_json['deployVersion']))
        tags=[]
        for ver in model_json['performance']:
            if ver['modelVersion']==model_json['deployVersion']:
                tags = tags + [model_json['outputTag']]+ver['inputTag']
        logger.info(f'all tags@@@@@@@ {tags}')
        startTime= startTime.strftime('%d-%m-%Y %H:%M')
        endTime= endTime.strftime('%d-%m-%Y %H:%M')
        timequery = {"type":'date',"start":startTime,"end":endTime}
    except Exception as e0:
        model_json['RunHistoricProgress'].append('Some issue with deployed model version')
        meta.updateModel(model_json,unitId)
        exit()
    final_dict, output_tag = pred(model_json, unitId, modelId, tags, timequery)
    #return final_dict, output_tag  
    if final_dict != {} or len(output_tag) != 0:
        tag_prefix1 = "pred_"
        tag_prefix2 = "flagModel__"
        outputTag1 = tag_prefix1 + output_tag
        outputTag2 = tag_prefix2 + output_tag
        predLwLabel = "predLw_" + output_tag
        predUpLabel = "predUp_" + output_tag
        post_tags = [outputTag1,outputTag2,predLwLabel,predUpLabel]
        logger.info(f'{outputTag1} {outputTag2} {predLwLabel} {predUpLabel}')
        cc=1
        try:
            for i in post_tags:
                try:
                    post_data=[]
                    for key, value in final_dict.items():
                        post_data.append([key, value[cc]])
                    postDataApi(i, post_data)
                    logger.info(f'{i} posting done of {len(post_data)}')
                    model_json['RunHistoricProgress'].append(f'{i} tag data published')
                    meta.updateModel(model_json,unitId)
                    cc+=1
                except:
                    logger.info(f'{i} tag data not published')
                    model_json['RunHistoricProgress'].append(f'{i} tag data not published')
                    meta.updateModel(model_json,unitId)
                    cc+=1
            model_json['RunHistoricProgress'].append(f'All tags data is pulished to kairos.')
            meta.updateModel(model_json,unitId)
            return
        except:
            pass
    else:
        logger.info('No able to get values.')
        model_json['RunHistoricProgress'].append(f'No values calculated')
        meta.updateModel(model_json,unitId)
        exit()

try:
    modelId = os.environ["MODEL_ID"]
except Exception as e:
    logger.info("required model Id")

try:
    startTime = os.environ["START_TIME"]
    endTime = os.environ["END_TIME"]
except:
    logger.info('required time period')
    
'''modelId='6389b47e96e3470007432946'
unitsId = '5f608d3a10723ca5deaab563' 
startTime=datetime.datetime.strptime("2023-06-07T12:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
#startTime= startTime.strftime('%d-%m-%Y %H:%M')

#convert endTime
endTime=datetime.datetime.strptime("2023-06-16T12:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')'''
#endTime= endTime.strftime('%d-%m-%Y %H:%M')
try:  
    calculation(modelId, startTime, endTime)
except Exception as E0:
    logger.info('Something went wrong')
    logger.info(E0)
finally:
    logger.info('Completed')