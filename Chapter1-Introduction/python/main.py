from simulation import *
from visualization import *

def main():
    y_true,y_estimated,errors = simulation(100)#进行100次仿真
    plot_y(y_true,y_estimated)
    plot_compare_y(y_true,y_estimated)
    plot_metric(errors,y_true)
    [RMSE,MAE,MAPE] = error_quantification(errors)
    print("RMSE:",RMSE," MAE:",MAE," MAPE:",MAPE,"%")


if __name__=='__main__':
    main()
