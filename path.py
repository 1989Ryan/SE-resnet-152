train_path = './datasets/train/'
test_path = './datasets/test/'
with open('test.txt', 'r') as test:
	with open('newtest.txt', 'w') as newtest:
		for line in test:
			newtest.write(test_path + line)
with open('train.txt', 'r') as train:
	with open('newtrain.txt', 'w') as newtrain:
		for line in train:
			newtrain.write(train_path + line)
