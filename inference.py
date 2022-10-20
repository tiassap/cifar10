import matplotlib.pyplot as plt
from load_cifar_10_pytorch import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Using test set only
cifar_10_dir = 'cifar-10-batches-py' 
(_, _), (x_test, y_test) = load_data(cifar_10_dir)

classname = ["airplane", "automobile","bird","cat","deer", "dog","frog", "horse","ship", "truck"]

# Preprocess dataset for inference
x, _ = preprocess(x_test, y_test)

# Load trained model
model = torch.load('trained_model.pt')
model = model.to(device)
model.eval()

# Display image and prediction in 25x25 grid
num_plot = 5
fig, ax = plt.subplots(num_plot, num_plot)
fig.suptitle("Ground truth | prediction")
for m in range(num_plot):
	for n in range(num_plot):
		idx = np.random.randint(0, x_test.shape[0])
		ax[m, n].imshow(x_test[idx])
		x_ = x[idx].to(device).float()
		x_ = torch.unsqueeze(x_, axis=0)
		y = model(x_)
		y = int(torch.argmax(y, dim=1))
		ax[m, n].set_title("{} | {}".format(classname[int(y_test[idx])], classname[int(y)]), loc='center', fontsize=6, y=1.0, pad = 1)
		ax[m, n].get_xaxis().set_visible(False)
		ax[m, n].get_yaxis().set_visible(False)
fig.subplots_adjust(hspace=0.25)
fig.subplots_adjust(wspace=0)
plt.savefig('prediction_plot.png')
plt.show()