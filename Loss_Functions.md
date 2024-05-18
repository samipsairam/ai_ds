### MSE: Mean Squared Error 
#### Average squared distance between Actual and Predicted Value
```
def mean_squared_error(actual_value, predicted_value, number_of_observation):
    squared_difference = (actual_value - predicted_value)**2
    MSE = squared_difference/n
    return MSE
```

#### MAE
```
def mean_absolute_error(actual_value, predicted_value, number_of_observation):
    difference = actual_value-predicted_value
    MAE =  (difference if difference >= 0 else -(difference)) / number_of_observation
    return MSE
```

#### HUBER LOSS FUNCTION
```
def huber_loss_func(actual_value, predicted_value, hinge_loss):
    difference = actual_value - predicted_value
    hl_diff = (difference if difference >= 0 else -(difference)) / number_of_observation
    HUBER_LOSS_VAL=0
    
    if hl_diff <= hinge_loss:
        HUBER_LOSS_VAL = hl_diff/2
    else:
        HUBER_LOSS_VAL = hinge_loss(hl_diff - hinge_loss/2)
    return HUBER_LOSS_VAL
```    
    
    

    
    
