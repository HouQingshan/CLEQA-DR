from models.models_IQA import model_IQA
from models.models_CLS import model_CLS
from models.models_joint import model_joint
from models.models_SR import model_SR
from models.test_model import TestModel
def create_model(opt):
	model = None
	if opt.model == 'model_IQA':
		model = model_IQA()
	elif opt.model == 'model_CLS':
		model = model_CLS()
	elif opt.model == 'model_SR':
		model = model_SR()
	elif opt.model == 'model_joint':
		model = model_joint()
	elif opt.model == 'test':
		model = TestModel()
		
	model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model