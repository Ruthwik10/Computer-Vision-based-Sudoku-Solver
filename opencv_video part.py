import torch
from Helper import *
from Model import CNN
import Sudoku_Solver

rec = cv.VideoCapture(0);
fourcc =cv.VideoWriter_fourcc(*'XVID');
out = cv.VideoWriter('result.avi',fourcc,40,(640,480));
rec.set(3,640)
rec.set(4,480)
print(rec.get(3),rec.get(4))
device = torch.device('cpu')
model = CNN().to(device)
model.load_state_dict(torch.load("models/final1"))
model.to(device)
while(True):
    r, frame = rec.read()
    processed = Preprocess(frame)
    mat,segmented,imgs = Segmentation(processed, frame)
    if segmented is not None:
        grid = np.zeros((81,50,50),dtype=np.uint8)
        for i in range(81):
            grid[i] = imgs[i,8:58,8:58]
        grid = torch.from_numpy(grid).unsqueeze(1).type(torch.FloatTensor)
        grid = grid.to(device)
        output = model(grid)
        _,pred = output.max(dim=1)
        pred = pred.to(torch.device('cpu')).numpy()
        pred1 = pred.reshape((9,9))
        #print(pred1)
        try:
            sudoku = Sudoku_Solver.fun(pred1.copy())
        except:
            pass
        if sudoku is not None:
            sudoku = sudoku.reshape((1,81))
            #print(sudoku)
            pred = pred.reshape((1,81))
            bool = np.where(pred>0,0,1)
            prnt_val =bool*sudoku
            id = np.nonzero(prnt_val)
            prnt_val = prnt_val[id]
            id,val_idx = np.nonzero(bool)
            for j, i in enumerate(val_idx):
                c = i % 9
                r = i // 9
                cv.putText(segmented, str(prnt_val[j]), (c * 66 + 20, r * 66 + 38), cv.FONT_HERSHEY_COMPLEX, 1,
                           (25, 10, 220), 3)
                matrix = cv.getPerspectiveTransform(np.float32([[0, 0], [594, 0], [0, 594], [594, 594]]),mat)
                mask = np.zeros((frame.shape))
                inv = cv.warpPerspective(segmented,matrix,(640,480))
                inv_gray = cv.cvtColor(inv,cv.COLOR_BGR2GRAY)
                ret, inv_gray = cv.threshold(inv_gray,20,255,cv.THRESH_BINARY)
                inv_not = cv.bitwise_not(inv_gray)
                inv_not = cv.bitwise_and(frame,frame,mask= inv_not)
                dst = cv.bitwise_xor(inv_not,inv )
                cv.imshow("frame",dst)
                out.write(dst)
        else:
            cv.imshow("frame",frame)
            out.write(frame)
    else:
        cv.imshow("frame", frame)
        out.write(frame)

    if cv.waitKey(1) & 0xFF == 27 :
        break

rec.release()
out.release()
cv.destroyAllWindows()