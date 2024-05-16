import sys
sys.path.append('.')

import xnat
import os
import tqdm
import pandas as pd
import shutil
import json
import argparse
import re

from abc import ABC, abstractmethod
from functools import wraps

from Core.Utils.Data_Aug import data_aug

###########XnatConnector###########
class XnatConnector:
    def __init__(self,URL,user,passwd) -> None:
        """
        arg:
            URL: URL of XNAT
            user: username
            password: password
            
        """
        self.URL = URL
        self.user = user
        self.passwd = passwd
        self.download_success = {}
        
    
    def connect_session(self,project):
        '''
        Start a session with XNAT
        arg:
            URL: URL of XNAT
            user: username
            password: password
            project name

        return:   
            XNAT project object
        '''
        session = xnat.connect(self.URL, user=self.user, password=self.passwd)
        proj = session.projects[project]
        
        return proj
###########DataDownloader###########

def check_path(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        output_path = kwargs.get('output_path')
        if output_path:
            if os.path.exists(output_path):
                print(f'Output path: {output_path} exists!')
            else:
                os.makedirs(output_path)
                print(f'Output path: {output_path} created!')
        return method(*args, **kwargs)
    return wrapper

def check_usable(experiment,scan,modality,quality):
        '''
        Check if the data is usable
        '''
        #only quality usable and right modality 
        if scan.quality == quality and experiment.label.startswith(modality):
            
            return True
        else:
            print('false!',quality,modality,scan.id)
            return False



class DataDownloadStrategy(ABC):
    def __init__(self,xnat_project=None) -> None:
        self.xnat_project = xnat_project

    @abstractmethod
    def download(self):
        pass

class DownloadFromCSV(DataDownloadStrategy):
    def __init__(self,xnat_project=None,data_csv=None) -> None:
        super().__init__(xnat_project)
        self.download_recorder = DownLoadRecord()
        self.data_csv = data_csv

    @check_path
    def download(self,format='NIFTI',modality='CT',quality='usable',output_path=None):
        """
        args:
            csv_file: csv file with the data to download
            format: format of the data to download
            output_path: path to store the downloaded data
        """
        try:
            download_data = pd.read_csv(self.data_csv)
            download_data = download_data.sort_values(by=['Experiment']) # sort by Experiment
            #download
            for i in range(download_data.shape[0]):
                
                subject = self.xnat_project.subjects[str(download_data.loc[i,'Subject'])]
                experiment = subject.experiments[download_data.loc[i,'Experiment']]
                scan = experiment.scans[str(download_data.loc[i,'Scan'])]

                subject_name,experiment_name,scan_name = subject.label,experiment.label,scan.id
                print(f"this is experiment:{experiment.label}, scan:{scan.id}")

                if check_usable(experiment,scan,modality,quality):
                    print('downloading!')
                    try:
                        scan.resources[format].download_dir(output_path)
                    except Exception as e:
                        self.download_recorder.record_failing_download(output_path,experiment_name,subject_name,scan_name)
        except KeyboardInterrupt:
            print('Download stopped by user!')
        
        finally:
        #write json
            with open(output_path + 'download_failing.json','w') as f:
                json.dump(self.download_recorder.failing_dic,f,indent=2)
    
class DownloadFromXnat(DataDownloadStrategy):
    def __init__(self,xnat_project=None,earliest=False) -> None:
        super().__init__(xnat_project)
        self.download_recorder = DownLoadRecord()
        self.earliest = earliest
    
    @check_path
    def download(self,format='NIFTI',modality='CT',quality='usable',output_path=None):
        #check data
        subjects = self.xnat_project.subjects.values()
        try:
            for subject in subjects:
                #if only download the first scan of each experiment
                if self.earliest:
                    experiment = subject.experiments[0]
                    scan = experiment.scans[0]

                    if check_usable(experiment,scan,modality,quality):
                        #avoid download error
                        try:
                            scan.resources[format].download_dir(output_path)
                        except Exception as e:
                            self.download_recorder.record_failing_download(subject.label,experiment.label,scan.id)
                    else:
                        self.download_recorder.record_failing_download(subject.label,experiment.label,scan.id)
                #otherwise download all data!
                else:
                    for experiment in subject.experiments.values():
                        for num,scan in enumerate(experiment.scans.values()):
                            if check_usable(experiment,scan,modality,quality):
                                #avoid download error
                                try:
                                    scan.resources[format].download_dir(output_path)

                                except Exception as e:
                                    self.download_recorder.record_failing_download(subject.label,experiment.label,scan.id)
                            else:
                                self.download_recorder.record_failing_download(subject.label,experiment.label,scan.id)
        except KeyboardInterrupt:
            print('Download stopped by user!')
        #write json
        finally:
            print(self.download_recorder.failing_dic)
            with open(output_path + 'download_failing.json','w') as f:
                json.dump(self.download_recorder.failing_dic,f,indent=2)

class DataDownloader:
    def __init__(self,strategy:DataDownloadStrategy) -> None: 
        self.strategy = strategy
    
    def download(self,*args,**kwargs):
        self.strategy.download(*args,**kwargs)



class DownLoadRecord:
    """
    Recored failing downloaded samples.
    One subject(patient) could have multiple experiments, and each experiment could have multiple scans.
    """
    def __init__(self) -> None:
        self.__failing_dic = {}

    def record_failing_download(self,subj_name,exp_name,scan_name=None):
        print(subj_name,exp_name,scan_name,666)
        if self.__failing_dic.get(subj_name) is None:
            self.__failing_dic[subj_name] = {exp_name:[scan_name]}
        else:
            self.__failing_dic[subj_name][exp_name].append(scan_name)
    
    @property
    def failing_dic(self):
        return self.__failing_dic
###############       
        
class Data_Uploade:
    def __init__(self,data_path,xnat_project) -> None:
        self.data_set = data_path
        self.xnat_project = xnat_project

    def upload_data(self,upload_path):
        """
        Upload data to the xnat project
        args:
            upload_path: path to the data to be uploaded
        """
        #check if the path exists
        upload_path_lst = [os.path.join(upload_path,file) for file in os.listdir(upload_path) if file.endswith('.nii.gz')]
        upload_data = pd.read_csv(self.data_set)
        upload_data = upload_data.sort_values(by=['Experiment'])


        for i in range(upload_data.shape[0]):
            subject = self.xnat_project.subjects[upload_data.loc[i,'Subject']]
            experiment = subject.experiments[upload_data.loc[i,'Experiment']]
            scan = experiment.scans[upload_data.loc[i,'Scan']]

            subject_name,experiment_name,scan_name = subject.label,experiment.label,scan.id
            print(f"this is experiment:{experiment.label}, scan:{scan.id}")
            try:
                scan.create_resource(label='nnUnet_0')
                scan.resources['nnUnet_0'].upload(upload_path_lst[i], os.path.basename(upload_path_lst[i]))
            except Exception as e:
                print('resource already exists',experiment_name,scan_name)
                scan.resources['nnUnet_0'].upload(upload_path_lst[i], os.path.basename(upload_path_lst[i]))
        #experiment.scans['4'].create_resource(label='nnUnet_0')
        #experiment.resources['nnUnet_0'].upload('../../Test_Data/CT_Phase/CILM_CT_100330.nii.gz', os.path.basename('../../Test_Data/CT_Phase/CILM_CT_100330.nii.gz'))
        #['nnUnet_0'].upload('../../Test_Data/CT_Phase/CILM_CT_100330.nii.gz', os.path.basename('../../Test_Data/CT_Phase/CILM_CT_100330.nii.gz'))
        experiment.scans['4'].resources['nnUnet_0'].upload('../../Test_Data/CT_Phase/CILM_CT_100330.nii.gz', os.path.basename('../../Test_Data/CT_Phase/CILM_CT_100330.nii.gz'))
        
class DataExtract:
    """
    Extract Data after downloading from xnat
    """
    file_suffix = '/resources/NIFTI/files/image.nii.gz'
    def __init__(self,Xnat_path) -> None:
        self.Xnat_path = Xnat_path

    @check_path
    def extract_data(self,project_name,output_path=None):
        """
        Extract data from the xnat folder
        """
        #check if the path exists
        experi_lst = os.listdir(self.Xnat_path)

        for exp_num,exp in enumerate(experi_lst):
            if exp.startswith('CT'):
                scans_lst = os.listdir(self.Xnat_path + exp + "/scans/")
                scans_id_lst = [scan.split('-')[0] for scan in scans_lst] # get a unique id for each scan
                for scan_id, scan in zip(scans_id_lst,scans_lst):
                    #experiment_str = str(experiment).split('_')[1]
                    file_name = project_name + '_' + exp + '_' + scan_id + '_' +  '0000' + '.nii.gz'
                    xnat_scan_path = self.Xnat_path + exp + '/scans/' + scan  + self.file_suffix
                    #move the file to the nnUNet folder
                    print(f"this is file name and snat path {file_name},{xnat_scan_path}")
                    shutil.copy(xnat_scan_path,output_path  + file_name)



    
class NNunetFormat:
    file_suffix = '/resources/NIFTI/files/image.nii.gz'

    def __init__(self,input_path,out_path,task_name) -> None:
        """
        arg:input_path: path to the xnat downloaded data
            out_path: path to the nnUNet folder with task name
            task_name: name of the task for the nnU-Net dataset eg:Task_503_CILM
        """
        self.input_path = input_path
        self.task = task_name
        
        # Verify task_name format
        if not re.match(r"Task_50\d_.+", task_name):
            raise ValueError("task_name must be in the format Task_50x_XXX where x is a digit and XXX is a string")
        
        self.nnunt_out_path = out_path + self.task 

    def move_files(self,project_name):
        '''
        Make the file name for the nnUNet
        arg:
            project_name: name of the project eg:CILM
        '''
        #get all the xnat downloaded files
        extractor = DataExtract(self.input_path)
        extractor.extract_data(project_name,output_path=self.nnunt_out_path + "/imagesTr/")#extract and move the files to the nnUNet folder

    def make_json_file(self):
        '''
        Function to make the json file for the nnUNet. 
        arg:
            output_path: path to save the folder
            folder_name: has to be in fromat (task_50{task number}_{name of the task})
        '''
        # get of the file names of the scans

# fill a list with dicts containing image file name and label file name
# when running infrence the label file does not exist, nnUNet still need it (I think?)
        list_of_dicts = []
        for scan_file in os.listdir(self.nnunt_out_path):
            dict = {"image":"./imagesTr/"+scan_file,"label":"./labelsTr/"+scan_file}
            list_of_dicts.append(dict)


        # boilerplate json file
        data = {
            "name": f"{self.task}",
            "description": f"{self.task}",
            "reference": "Erasmus Medical Centre",
            "licence": "Not applicable",
            "release": "Not applicable",
            "tensorImageSize": "3D",
            "modality": {"0": "3D"
            },

            "labels": {
                "0": "background",
                "1": "organ",
                "2": "tumor"
            },
            "numTraining": 10,
            "numTest": 0,
            "training": list_of_dicts,# add list of dicts


        "test": []
        }
        
        # save the json file
        with open(self.nnunt_out_path + '/dataset.json', 'w') as outfile:
            json.dump(data, outfile, indent=2)




def parse_args():
    parser = argparse.ArgumentParser(description="Download and format data for nnU-Net from XNAT")
    subparsers = parser.add_subparsers(dest='command')

###For downloading data from XNAT
    parser_download = subparsers.add_parser('Download',help='Download data from XNAT')
    parser_download.add_argument('--url', type=str, required=False, default='https://bigr-rad-xnat.erasmusmc.nl',help='The URL of the XNAT instance')
    parser_download.add_argument('--user', type=str, required=False, default='yliu',help='Username for XNAT')
    parser_download.add_argument('--passwd', type=str, required=False, default='x37vnp78',help='Password for XNAT')
    parser_download.add_argument('--project', type=str, required=True, help='The project name in XNAT')
    parser_download.add_argument('--download_type',help='Download from csv or all data from XNAT',choices=['csv','xnat'],default='csv')
    parser_download.add_argument('--data_csv', type=str, default=None,help='CSV file containing data to be downloaded')
    parser_download.add_argument('--xnat_type',choices=['earliest','all'],help='Whether to download earliest or all data from XNAT')
    parser_download.add_argument('--store_out_path', type=str, required=True, help='The path where the downloaded data will be stored')
    
    parser_extract = subparsers.add_parser('Extract',help='Extract data from XNAT') 
    parser_extract.add_argument('--Xnat_path', type=str, required=True, help='The path to the XNAT data')
    parser_extract.add_argument('--out_path', type=str, required=True, help='The path where the extracted data will be stored')
    parser_extract.add_argument('--project_name', type=str, required=True, help='The name of the project')

    parse_nnunet = subparsers.add_parser('nnunet',help='create folder and json file for nnU-Net')
    parse_nnunet.add_argument('--Xnat_path', type=str, required=True, help='The path to the XNAT data')
    parse_nnunet.add_argument('--out_path', type=str, required=True, help='The path where the nnU-Net data will be stored')
    parse_nnunet.add_argument('--task_name', type=str, required=True, help='The name of the task for the nnU-Net dataset')
    parse_nnunet.add_argument('--project_name', type=str, required=True, help='The name of the project')

    return parser.parse_args()

    


if __name__ == "__main__":
 
    def main():
        args = parse_args()
        # Now you can use args.url, args.user, args.passwd, etc., in your script.

        # Establish the connection to XNAT and see whether we need to download data
        if args.command == 'Download':
            xnat_session = XnatConnector(args.url, args.user, args.passwd)
            project = xnat_session.connect_session(args.project)
            if args.download_type == 'csv':
                # Download data from XNAT using the supplied data
                download_strategy = DownloadFromCSV(project,args.data_csv)
            elif args.download_type == 'xnat':
                # Download all data from XNAT
                download_strategy = DownloadFromXnat(project,args.xnat_type)
            downloader = DataDownloader(download_strategy)
            downloader.download(format='NIFTI',modality='CT',quality='usable',output_path=args.store_out_path) # set desired download format
        
        elif args.command == 'Extract':
            # Extract data from XNAT
            extract_data = DataExtract(args.Xnat_path)
            extract_data.extract_data(args.project_name,args.out_path)

        elif args.command == 'nnunet':
            # Create folder and json file for nnU-Net
            nnunet_format = NNunetFormat(args.Xnat_path,args.out_path,args.task_name)
            nnunet_format.move_files(args.project_name)
            nnunet_format.make_json_file()
        
        elif args.command == 'upload':
            pass



        if args.upload_path:
            # Upload the data to XNAT
            xnat_session = XnatConnector(args.url, args.user, args.passwd)
            project = xnat_session.connect_session(args.project)
            data_upload = Data_Uploade(args.data_csv,project)
            data_upload.upload_data(args.upload_path)
    main()



