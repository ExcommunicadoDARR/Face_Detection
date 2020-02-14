import matplotlib.pyplot as plt

def plot(history,e):

    train_loss=history.history['loss']
    train_acc=history.history['acc']
    val_loss=history.history['val_loss']
    val_acc=history.history['val_acc']
    print(train_loss,val_loss)
    epochs=range(0,e)
    fig=plt.figure()
    fig.suptitle("Model")
    plt.ylim(0.0,2.0)
    plt.xlim(0,10)
    plt.plot(epochs,train_loss,'r',label="training_loss")
    #plt.plot(epochs,train_acc,'g',label="training_acc")
    plt.plot(epochs,val_loss,'b',label="validation_loss")
    #plt.plot(epochs,val_acc,'m',label="validation_acc")    
    
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
#print(model.summary())
