import copy
import matplotlib.pyplot as plt

from IPython.display import clear_output


def rectify_labels(X, y, i_start=0, max_iters=None):
    """
    A tool for manual labeling in Jupyter Notebook
    """
    fig, ax = plt.subplots()

    y_upd = copy.deepcopy(y)
    img = ax.imshow(X[0], cmap='gray')
    break_flag = False
    
    try:
        for i in range(i_start, len(y)):
            img.set_data(X[i])
            ax.set_title(f'y={y[i]}')
            fig.canvas.draw()
            res = input().strip()
            if res:
                if int(res) in [0, 1]:
                    y_upd[i] = int(res)
                else:
                    y_upd[i] = 1
                    print('Considering input as 1')

            if ((max_iters is not None)
                and (i - i_start >= max_iters-1)):
                break_flag = True
                break

    except KeyboardInterrupt: pass
    finally: return i+1, break_flag, y_upd


def mend_labels(X, y, clear_after=5):
    """
    Wrapper over `rectify_labels`
    preventing the input bar from running far down 
    """
    i_end = 0
    y_upd = copy.deepcopy(y)
    break_flag = True
    

    while break_flag:
        i_end, break_flag, y_upd = rectify_labels(
            X, y_upd, i_end, max_iters=clear_after)
        clear_output()

    return y_upd
