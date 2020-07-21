from lib.TenFoldCrossValidation import TenFoldCrossValidation



# Feel free to modify this stuff
sets = ["data/clean_dataset.txt", "data/noisy_dataset.txt"]
verbose_logging = True




# But please leave this stuff as is
tfcv = TenFoldCrossValidation()
tfcv.run(sets, logging=verbose_logging)
