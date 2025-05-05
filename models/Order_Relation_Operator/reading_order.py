# reading_order.py
import torch, numpy as np
from models.Order_Relation_Operator.src.models import MLP
from models.Order_Relation_Operator.src.decode import CountDecoder, GreedyDecoder, HamiltonianDecoder, ForwardBackwardDecoder

CATEGORY_BITS = 6          # FCR = 6 types
LINE_FEAT     = CATEGORY_BITS + 6        # 12

def _feat(box, W, H):
    x0,y0,x1,y1 = box
    cx, cy = ((x0+x1)/2)/W, ((y0+y1)/2)/H
    lx, ly =  x0/W,  y1/H
    rx, ry =  x1/W,  y1/H
    onehot = np.zeros(CATEGORY_BITS, np.float32)
    onehot[0] = 1.0
    return np.concatenate([onehot, [cx,cy,lx,ly,rx,ry]]).astype(np.float32)

def order_boxes(rects, img_shape, model_path,decoder="count"):
    H, W   = img_shape[:2]
    feats  = np.stack([_feat(b,W,H) for b in rects])   # shape (n,12)
    n      = len(rects)
    pairs  = np.concatenate([np.repeat(feats, n, 0),
                             np.tile   (feats, (n,1))], 1)   # (n², 24)

    in_size  = pairs.shape[1]         # 24  ← correspond au checkpoint
    hid_size = in_size * 2            # 48
    net = MLP(in_size, hid_size, 1)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()

    with torch.no_grad():
        P = torch.sigmoid(net(torch.from_numpy(pairs))).view(n,n).numpy()

     # --- Choix du décodeur ---
    decoder = decoder.lower()
    if decoder == "count":
        dec = CountDecoder(P)
    elif decoder == "greedy":
        dec = GreedyDecoder(P)
    elif decoder == "hamiltonian":
        dec = HamiltonianDecoder(P)
    elif decoder == "forwardbackward":
        dec = ForwardBackwardDecoder(P)
    else:
        raise ValueError(f"Unknown decoder: {decoder}")

    dec.run()
    return dec.best_path