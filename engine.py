import numpy as np
import torch
from torch import nn
import torch.optim as optim
from feedforward import FeedForwardNet, NetConv, Net
from state_machine import StateMachine
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation import evaluate, evaluate_others

state = StateMachine()

def inference():
    net.compute_cluster_center(alpha)
    net.eval()
    feature_vector = []
    labels_vector = []
    pred_vector = []
    with torch.no_grad():
        for step, (x, y) in enumerate(state.dataset.loader_wrapper["test"]):
            with torch.no_grad():
                z = net.encode(x)
                pred = net.predict(z)
            feature_vector.extend(z.detach().cpu().numpy())
            labels_vector.extend(y.numpy())
            pred_vector.extend(pred.detach().cpu().numpy())
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    pred_vector = np.array(pred_vector)
    return feature_vector, labels_vector, pred_vector


def visualize_cluster_center():
    with torch.no_grad():
        cluster_center = net.compute_cluster_center(alpha)
        reconstruction = net.decode(cluster_center)

    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(
            reconstruction[i]
            .detach()
            .cpu()
            .numpy()
            .reshape(state.dataset[0][0].shape[1], state.dataset[0][0].shape[2]),
            cmap="gray",
        )
    plt.savefig("./cluster_center.png")
    plt.close()

# Create an instance of the model
#model = FeedForwardNet(input_size=3, hidden_sizes=[64, 128], num_classes=state.dataset.get_num_classes())

net = Net(dim=3, class_num=state.dataset.get_num_classes())
optimizer = torch.optim.Adadelta(net.parameters())
criterion = nn.MSELoss(reduction="mean")

start_epoch = 0
epochs = 3001
alpha = 0.001
net.normalize_cluster_center(alpha)


for epoch in range(start_epoch, epochs):
    loss_clu_epoch = loss_rec_epoch = 0
    net.train()
    for step, (x, y) in enumerate(state.dataset.loader_wrapper["train"]):
        z = net.encode(x)

        if epoch % 2 == 1:
            cluster_batch = net.cluster(z)
        else:
            cluster_batch = net.cluster(z.detach())
        soft_label = F.softmax(cluster_batch.detach(), dim=1)
        hard_label = torch.argmax(soft_label, dim=1)
        delta = torch.zeros((state.batch_size, 10), requires_grad=False)
        for i in range(state.batch_size):
            delta[i, torch.argmax(soft_label[i, :])] = 1
        loss_clu_batch = 2 * alpha - torch.mul(delta, cluster_batch)
        loss_clu_batch = 0.01 / alpha * loss_clu_batch.mean()

        x_ = net.decode(z)
        loss_rec = criterion(x, x_)

        loss = loss_rec + loss_clu_batch
        optimizer.zero_grad()
        loss.backward()
        if epoch % 2 == 0:
            net.cluster_layer.weight.grad = (
                F.normalize(net.cluster_layer.weight.grad, dim=1) * 0.2 * alpha
            )
        else:
            net.cluster_layer.zero_grad()
        optimizer.step()
        net.normalize_cluster_center(alpha)
        loss_clu_epoch += loss_clu_batch.item()
        loss_rec_epoch += loss_rec.item()
    print(
        f"Epoch [{epoch}/{epochs}]\t Clu Loss: {loss_clu_epoch / len(state.dataset.loader_wrapper["train"])}\t Rec Loss: {loss_rec_epoch / len(state.dataset.loader_wrapper["train"])}"
    )

    if epoch % 50 == 0:
        visualize_cluster_center()
        feature, label, pred = inference()
        nmi, ari, acc = evaluate(label, pred)
        print("Model NMI = {:.4f} ARI = {:.4f} ACC = {:.4f}".format(nmi, ari, acc))
        ami, homo, comp, v_mea = evaluate_others(label, pred)
        print(
            "Model AMI = {:.4f} Homogeneity = {:.4f} Completeness = {:.4f} V_Measure = {:.4f}".format(
                ami, homo, comp, v_mea
            )
        )
        #save_model(net, optimizer, epoch)






# Assume 'labels_encoded' is your numpy array of labels
#classes, class_counts = np.unique(state.dataset.labels, return_counts=True)
#criterion = nn.CrossEntropyLoss(weight=class_weights)
#class_counts = torch.tensor(class_counts, dtype=torch.float)
# Calculate class weights inversely proportional to class counts
#class_weights = 1.0 / class_counts
# Normalize the weights (optional)
#class_weights = class_weights / class_weights.sum()
#criterion = nn.CrossEntropyLoss(weight=class_weights)

#optimizer = optim.SGD(model.parameters(), lr=0.001)

# Define the evaluation function

# Train the model
#num_epochs: int = 400  # Adjust as needed
#State.train_model(net, state.dataset.loader_wrapper["train"], criterion, optimizer, num_epochs)

# Evaluate the model
# Get class names from the label encoder
#class_names = ["0","1","2","3"]
#accuracy, cm = State.evaluate_model(net, state.dataset.loader_wrapper["test"], class_names=class_names)

# Save the model
#torch.save(model.state_dict(), 'feedfwdnet.pth')

# Load the model
#model = SimpleFeedforwardNet(input_size=inputs.shape[1], hidden_sizes=[15, 20], num_classes=num_classes)
#model.load_state_dict(torch.load('model.pth'))
#model.eval()