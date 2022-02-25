class Config:
    #solve
    BATCH_SIZE = 32
    PROPOSAL_NUM = 6
    CAT_NUM = 4
    INPUT_SIZE = (448, 448)  # (w, h)
    #optimizer
    MAX_EPOCH=150
    STEPS = [60, 100]
    LR = 0.001
    WD = 1e-4
    #eval & save
    SAVE_FREQ = 10
    cuda_id = '0,1'
    resume = ''
    test_model = 'model.ckpt'
    save_dir = './models/'
