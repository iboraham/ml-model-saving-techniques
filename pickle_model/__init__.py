import pickle

def pickle_model(model=None, save=False, filepath='model.pkl'):
    if save:
        pickle.dump( model, open( filepath, "wb" ))
    else:
        saved_model = pickle.load(open( filepath, "rb" ))
        return saved_model
