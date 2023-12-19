"""
This script aims to download correct nnUnet data format from XNAT

It has several options:
    ---1. Download per patient with 
    ---2. Download data with given IDs
"""
import xnat
import os
import tqdm
import pandas as pd
import shutil
import json
import argparse

class XnatSession:
    def __init__(self,URL,user,passwd) -> None:
        self.URL = URL
        self.user = user
        self.passwd = passwd
        self.download_success = {}
        
    
    def start_xnat_session(self,project):
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

class Data_Download:
    def __init__(self,data=None,xnat_project=None):
        """
        arg:
            data: csv file with the data to download
            xnat_project: xnat project object       
        """
        self.xnat_project = xnat_project
        self.__download_data = data
        self.__suceess_lst = {}

        print('this is the project:',self.xnat_project)


    def download_from_data(self,output_path,format='NIFTI'):
        """
        Download data from supplied data.
        """
        #check if the path exists
        self._check_path(output_path)

        download_data = pd.read_csv(self.__download_data)
        download_data = download_data.sort_values(by=['Experiment'])


        for i in range(download_data.shape[0]):
            
            subject = self.xnat_project.subjects[str(download_data.loc[i,'Subject'])]
            experiment = subject.experiments[download_data.loc[i,'Experiment']]
            scan = experiment.scans[str(download_data.loc[i,'Scan'])]

            subject_name,experiment_name,scan_name = subject.label,experiment.label,scan.id
            print(f"this is experiment:{experiment.label}, scan:{scan.id}")
            try:
                self._download_single_image(experiment,scan,output_path)
                self._store_download_success(output_path,experiment_name,subject_name,scan_name)
            except Exception as e:
                self._recording_failing_download(output_path,experiment_name,subject_name,scan_name)
        
        with open(output_path + 'download_success.json','w') as f:
            json.dump(self.__suceess_lst,f,indent=2)




    def down_load_from_xnat(self,output_path,earliest=False,format='NIFTI'):
        #check data
        self._check_path(output_path)
        subjects = self.xnat_project.subjects.values()
        for subject in subjects:
            if earliest:
                experiment = subject.experiments[0]
                scan = experiment.scans[0]
                try:
                    self._download_single_image(experiment,scan,output_path)
                    self._store_download_success(output_path,experiment.label,subject.label)
                except Exception as e:
                    self._recording_failing_download(output_path,experiment.label,subject.label)
                #otherwise download all data!
            else:
                for experiment in subject.experiments.values():
                    for num,scan in enumerate(experiment.scans.values()):
                        print('fuck')
                        try:
                            print(f"this is experiment:{experiment.label}, scan:{scan.id}")
                            self._download_single_image(experiment,scan,output_path,format)
                            self._store_download_success(output_path,experiment.label,subject.label,scan.id)
                        except Exception as e:
                            self._recording_failing_download(output_path,experiment.label,subject.label,scan.id)
        with open(output_path + 'download_success.json','w') as f:
            json.dump(self.__suceess_lst,f,indent=2)

    
    
    def _download_single_image(self,experiment,scan,output_path,modality='CT',quality='usable',format='NIFTI'):
        if self._check_usable(experiment,scan,modality,quality):
            print('downloading!')
            scan.resources[format].download_dir(output_path)
            


    def _recording_failing_download(self,output_path,subj_name,exp_name,scan_name=None):
        with open(output_path + 'download_failing_lst.txt','a') as f:
            f.write(f"{subj_name}_{exp_name}_{scan_name} +\n")

    def _store_download_success(self,output_path,subj_name,exp_name,scan_name=None):
        if self.__suceess_lst.get(subj_name) is None:
            self.__suceess_lst[subj_name] = {exp_name:[scan_name]}
        else:
            self.__suceess_lst[subj_name][exp_name].append(scan_name)
        print(self.__suceess_lst)






    def _check_path(self,output_path):
        if os.path.exists(output_path):
            print(f'Output path:{output_path} exists!')
        else:
            os.makedirs(output_path)
            print(f'Output path:{output_path} created!')
        


           
    def _check_usable(self,experiment,scan,modality,quality):
        '''
        Check if the data is usable
        '''
        #only quality usable and right modality 
        if scan.quality == quality and experiment.label.startswith(modality):
            
            return True
        else:
            print('false!',modality)
            print(experiment.label.startswith(modality),modality)
            return False
        
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
        upload_path_lst = [os.path.join(upload_path,file) for file in os.listdir(upload_path)]
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

    def extract_data(self,project_name,out_path):
        """
        Extract data from the xnat folder
        """
        #check if the path exists
        print(os.getcwd())
        print('fuck',self.Xnat_path)
        experi_lst = os.listdir(self.Xnat_path)
        self._path_check(out_path)

        for exp_num,exp in enumerate(experi_lst):
            if exp.startswith('CT'):
                scans_lst = os.listdir(self.Xnat_path + exp + "/scans/")
                for scan_num,scan in enumerate(scans_lst):
                    #experiment_str = str(experiment).split('_')[1]
                    file_name = project_name + '_' + exp + str(scan_num) + '_' +  '0000' + '.nii.gz'
                    xnat_scan_path = self.Xnat_path + exp + '/scans/' + scan  + self.file_suffix
                    #move the file to the nnUNet folder
                    print(f"this is file name and snat path {file_name},{xnat_scan_path}")
                    shutil.move(xnat_scan_path,out_path  + file_name)

    def _path_check(self,out_path):
        if os.path.exists(out_path):
            print(f'Output path:{out_path} exists!')
        else:
            os.makedirs(out_path)
            print(f'Output path:{out_path} created!')

    
class NNunetFormat:
    file_suffix = '/resources/NIFTI/files/image.nii.gz'

    def __init__(self,input_path,nnunt_out_path,task) -> None:
        """
        arg:input_path: path to the xnat downloaded data
            nnunt_out_path: path to the nnUNet folder with task name
        """
        self.input_path = input_path
        self.nnunt_out_path = nnunt_out_path
        self.task = task
        self.scan_path = self.nnunt_out_path + self.task + "/imagesTr/"
        # if out path not exist, make it
        if not os.path.exists(self.scan_path):
            os.makedirs(self.scan_path)
        else:
            print('nnUNet folder exists!',nnunt_out_path)
    
    def make_file_name(self,project_name):
        '''
        Make the file name for the nnUNet
        arg:
            project_name: name of the project eg:CILM
        '''
        #get all the xnat downloaded files
        experiment_lst = os.listdir(self.input_path)
        for exp_num,exp in enumerate(experiment_lst):
            if exp.startswith('CT'):
                scans_lst = os.listdir(self.input_path + exp + "/scans/")
                for scan_num,scan in enumerate(scans_lst):
                    #experiment_str = str(experiment).split('_')[1]
                    file_name = project_name + '_' + exp + str(scan_num) + '_' +  '0000' + '.nii.gz'
                    xnat_scan_path = self.input_path + exp + '/scans/' + scan  + self.file_suffix
                    #move the file to the nnUNet folder
                    print(f"this is file name and snat path {file_name},{xnat_scan_path}")
                    shutil.move(xnat_scan_path,self.nnunt_out_path + self.task + "/imagesTr/" + file_name)
        #self._delete_file()
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
        scan_path = self.nnunt_out_path + self.task + "/imagesTr/"
        list_of_dicts = []
        for scan_file in os.listdir(scan_path):
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
        with open(self.nnunt_out_path + self.task + '/dataset.json', 'w') as outfile:
            json.dump(data, outfile, indent=2)

    def _delete_file(self,file_path):
        """
        delete the file of the path
        """
        shutil.rmtree(file_path)




    


if __name__ == "__main__":
    # out_path = '/data/scratch/r098986/CT_Phase/Data/Raw_Phase_Data/'
    # xnat_session = XnatSession('https://bigr-rad-xnat.erasmusmc.nl','yliu','x37vnp78')
    # extract_data = DataExtract('/data/scratch/r098986/CT_Phase/Data/Raw_Phase_Data/')
    # extract_data.extract_data('CILM',out_path)
    # xnat_proj = xnat_session.start_xnat_session('CILM')
    # data_download = Data_Download(xnat_project=xnat_proj,data='/data/scratch/r098986/CT_Phase/Data/True_Label/Phase_label_all.csv')
    # data_download.download_from_data(out_path)

    # nnunet_format = NNunetFormat(out_path,'/data/scratch/r098986/nnUnet_Seg/nnUNet_raw_data_base/nnUNet_raw_data/','Task_503_HGCSorafenib_Data')
    # nnunet_format.make_file_name('CILM')
    # nnunet_format.make_json_file()
    #nnunet_format.delete_file()

    # out_path = './data/scratch/r098986/nnUnet_Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task501_CILM_Liver_Samuel/'
    # xnat_session = XnatSession('https://bigr-rad-xnat.erasmusmc.nl','yliu','x37vnp78')
    # xnat_proj = xnat_session.start_xnat_session('CILM')
    # xnat_session.download_data(out_path,xnat_proj,earliest=False,required_data='/trinity/home/r098986/CRLM_Yizhou/Test_Data/scans_used_all_info.csv)
    # nnunet_format = NNunetFormat(out_path,/data/scratch/r098986/nnUnet_Seg/nnUNet_raw_data_base/nnUNet_raw_data/,'Task501_CILM_Liver_Samuel')
    # nnunet_format.make_file_name('CILM')
    # nnunet_format.make_json_file()


    def parse_args():
        parser = argparse.ArgumentParser(description="Download and format data for nnU-Net from XNAT")
        parser.add_argument('--url', type=str, required=False, default='https://bigr-rad-xnat.erasmusmc.nl',help='The URL of the XNAT instance')
        parser.add_argument('--user', type=str, required=False, default='yliu',help='Username for XNAT')
        parser.add_argument('--passwd', type=str, required=False, default='x37vnp78',help='Password for XNAT')
        parser.add_argument('--project', type=str, required=True, help='The project name in XNAT')
        parser.add_argument('--Download_Data',action='store_true',help='Whether to download data')
        parser.add_argument('--store_out_path', type=str, required=False, help='The path where the downloaded data will be stored')
        parser.add_argument('--data_csv', type=str, default=None,help='CSV file containing data to be downloaded')
        parser.add_argument('--task_name', type=str, required=False, help='The task name for the nnU-Net dataset')
        parser.add_argument('--nnUnet_path',type=str,default=False,help='nnUnet path')
        parser.add_argument('--Xnat_path',type=str,default=None,help='The path to the XNAT data')
        parser.add_argument('--extract_out_path',type=str,default=None,help='The path to extract the data from XNAT')
        parser.add_argument('--format',default='NIFTI',help='Data format to download')
        parser.add_argument('--upload_path',type=str,default=None,help='The path to upload the data to XNAT')
        # Add more arguments as needed

        return parser.parse_args()

    def main():
        args = parse_args()
        # Now you can use args.url, args.user, args.passwd, etc., in your script.

        # Establish the connection to XNAT and see whether we need to download data

        if args.Download_Data:
            print(args.Download_Data,'shit')
            #print(args.Download_Data,'fuck')
            xnat_session = XnatSession(args.url, args.user, args.passwd)
            project = xnat_session.start_xnat_session(args.project)
            if args.data_csv:
                # Download data from XNAT using the supplied data
                data_download = Data_Download(data=args.data_csv, xnat_project=project)
                data_download.download_from_data(args.store_out_path,args.format)
            else:
                # Download all data from XNAT
                data_download = Data_Download(xnat_project=project)
                print(666666)
                data_download.down_load_from_xnat(args.store_out_path,args.format)
        
        # Format the data for nnU-Net
        if args.nnUnet_path:
            nnunet_format = NNunetFormat(args.store_out_path,args.nnUnet_path,args.task_name)
            nnunet_format.make_file_name(args.project)
            nnunet_format.make_json_file()
        elif args.extract_out_path:
            # Extract the data from XNAT
            extract_data = DataExtract(args.Xnat_path)
            extract_data.extract_data(args.project,args.extract_out_path)

        if args.upload_path:
            # Upload the data to XNAT
            xnat_session = XnatSession(args.url, args.user, args.passwd)
            project = xnat_session.start_xnat_session(args.project)
            data_upload = Data_Uploade(args.data_csv,project)
            data_upload.upload_data(args.upload_path)
    main()

    # Use other classes and functions with the provided arguments
    # ...


