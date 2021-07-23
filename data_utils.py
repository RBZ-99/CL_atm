import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from batch_data import BatchData
import torchvision.transforms.functional as TorchVisionFunc
from core50 import core50
from toybox import toybox
from ilab import ilab


def get_split_cifar100_tasks(num_tasks, batch_size,run,paradigm,dataset):
	"""
	Returns data loaders for all tasks of split CIFAR-100
	:param num_tasks:
	:param batch_size:
	:return:
	
	datasets = {}
	
	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_transforms)
	
	for task_id in range(1, num_tasks+1):
		train_loader, test_loader = get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets
	"""
	
	datasets = {}
	train_datasets = [[] for i in range(num_tasks)]
	test_datasets = [[] for i in range(num_tasks)]

	if dataset == 'core50':
		for task_id in range(0, num_tasks):
			train_loader, test_loader = dataset_core50(task_id,batch_size,run,paradigm,dataset) #get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
			train_datasets[task_id] = train_loader
			test_datasets[task_id] = test_loader
	
	if dataset == 'toybox':
		for task_id in range(0, num_tasks):
			train_loader, test_loader = dataset_toybox(task_id,batch_size,run,paradigm,dataset) #get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
			train_datasets[task_id] = train_loader
			test_datasets[task_id] = test_loader	
	if dataset == 'ilab':
		for task_id in range(0, num_tasks):
			train_loader, test_loader = dataset_ilab(task_id,batch_size,run,paradigm,dataset) #get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
			train_datasets[task_id] = train_loader
			test_datasets[task_id] = test_loader
	returning = [[],[]]
	returning[0] = train_datasets
	returning[1] = test_datasets
	#print("returning dataset")
	return returning


def dataset_ilab(task_id, batch_size,run,paradigm,dataset_name):
			test_xs = [[],[],[],[],[],[],[]]
			test_ys = [[],[],[],[],[],[],[]]
			#train_xs = [[],[],[],[],[],[]]
			#train_ys = [[],[],[],[],[],[]]
			input_transform= Compose([
									transforms.Resize(32),
									transforms.RandomHorizontalFlip(),
									transforms.RandomCrop(32,padding=4),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			input_transform_eval= Compose([
									transforms.Resize(32),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			test_accs = []

			dataset = ilab( paradigm, run)
			#print(f"Incremental num : {task_id}")
			train, val, test = dataset.getNextClasses(task_id)
			print(len(train), len(val), len(test))
			train_x, train_y = zip(*train)
			val_x, val_y = zip(*val)
			test_x, test_y = zip(*test)

			#if inc_i > 0 :
			#    epoches = 1 #stream learning; see data only once

			train_data = DataLoader(BatchData(train_x, train_y, dataset_name,input_transform),
						batch_size=batch_size, shuffle=True, drop_last=True)
			#val_data = DataLoader(BatchData(val_x, val_y, dataset,input_transform_eval),batch_size=batch_size, shuffle=False)            
			test_data = DataLoader(BatchData(test_x, test_y,dataset_name, input_transform_eval),
						batch_size=batch_size, shuffle=False)
			
			#print("returned the values")
			
			return train_data, test_data


def dataset_core50(task_id, batch_size,run,paradigm,dataset_name):
			test_xs = [[],[],[],[],[]]
			test_ys = [[],[],[],[],[]]
			train_xs = [[],[],[],[],[]]
			train_ys = [[],[],[],[],[]]
			input_transform= Compose([
									transforms.ToTensor(),
                                    transforms.ToPILImage(),
									transforms.Resize(32),
									transforms.RandomHorizontalFlip(),
									transforms.RandomCrop(32,padding=4),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			input_transform_eval= Compose([
									transforms.ToTensor(),
                                    transforms.ToPILImage(),
									transforms.Resize(32),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			test_accs = []
			#for inc_i in range(task_id+1):
				#paradigm = 'class_iid'
				#run = 0
			dataset = core50( paradigm, run)
			#print(f"Incremental num : {task_id}")
			train, val, test = dataset.getNextClasses(task_id)
			print(len(train), len(val), len(test))
			train_x, train_y = zip(*train)
			val_x, val_y = zip(*val)
			test_x, test_y = zip(*test)
				

			#if inc_i > 0 :
			#    epoches = 1 #stream learning; see data only once

			train_data = DataLoader(BatchData(train_x, train_y,dataset_name, input_transform),
						batch_size=batch_size, shuffle=True, drop_last=True)
			#val_data = DataLoader(BatchData(val_x, val_y, input_transform_eval),
						#batch_size=batch_size, shuffle=False)            
			test_data = DataLoader(BatchData(test_x, test_y,dataset_name, input_transform_eval),
						batch_size=batch_size, shuffle=False)
			

			return train_data, test_data

def dataset_toybox(task_id, batch_size,run,paradigm,dataset_name):
			test_xs = [[],[],[],[],[],[]]
			test_ys = [[],[],[],[],[],[]]
			#train_xs = [[],[],[],[],[],[]]
			#train_ys = [[],[],[],[],[],[]]
			input_transform= Compose([
									transforms.Resize(32),
									transforms.RandomHorizontalFlip(),
									transforms.RandomCrop(32,padding=4),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			input_transform_eval= Compose([
									transforms.Resize(32),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			test_accs = []

			dataset = toybox( paradigm, run)
			#print(f"Incremental num : {task_id}")
			train, val, test = dataset.getNextClasses(task_id)
			print(len(train), len(val), len(test))
			train_x, train_y = zip(*train)
			val_x, val_y = zip(*val)
			test_x, test_y = zip(*test)
				

			#if inc_i > 0 :
			#    epoches = 1 #stream learning; see data only once

			train_data = DataLoader(BatchData(train_x, train_y,dataset_name, input_transform),
						batch_size=batch_size, shuffle=True, drop_last=True)
			#val_data = DataLoader(BatchData(val_x, val_y,dataset, input_transform_eval),
						#batch_size=batch_size, shuffle=False)            
			test_data = DataLoader(BatchData(test_x, test_y,dataset_name, input_transform_eval),
						batch_size=batch_size, shuffle=False)
			

			return train_data, test_data



