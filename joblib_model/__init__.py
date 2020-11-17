from joblib import dump, load

def joblib_model(model=None, save=False, filepath='model.pkl'):
    if save:
        dump( model, open( filepath, "wb" ))
    else:
        saved_model = load(open( filepath, "rb" ))
        return saved_model
