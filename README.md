# Model Architecture and Parameters


This README format includes detailed explanations of how the parameters for each layer are calculated, making it easier to understand the architecture and the reasoning behind the parameter counts.


## Model Layers

1. **Input Layer with Batch Normalization**

    ```python
    model_o.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model_o.add(BatchNormalization())
    ```

    - **Dense Layer**
        - **Input Shape**: `input_dim = X_train.shape[1]`
        - **Output Shape**: `128`
        - **Parameters**: `128 * (input_dim + 1)`
        
        Each neuron has `input_dim` weights plus one bias. Thus, for 128 neurons:
      
        **Params** = `128 * (input_dim + 1)`

    - **Batch Normalization**
        - **Parameters**: `4 * 128`
        
        Batch Normalization has 4 parameters per neuron: 2 for scale and shift (trainable), and 2 for mean and variance (non-trainable).
      

        `Trainable Params` = `2 * 128` = 256
    
        `Non-Trainable Params` = `2 * 128` = 256
      

2. **Hidden Layer with Dropout and Batch Normalization**

    ```python
    model_o.add(Dense(64, activation='relu'))
    model_o.add(BatchNormalization())
    model_o.add(Dropout(0.5))
    ```

    - **Dense Layer**
        - **Input Shape**: `128`
        - **Output Shape**: `64`
        - **Parameters**: `64 * (128 + 1)`
        
        Each neuron has 128 weights plus one bias. Thus, for 64 neurons:
      
  
        **Params** = `64 * (128 + 1) `= 8256
       

    - **Batch Normalization**
        - **Parameters**: `4 * 64`
        
        Batch Normalization has 4 parameters per neuron: 2 for scale and shift (trainable), and 2 for mean and variance (non-trainable).
        
       **Trainable Params** = `2 * 64` = 128
    
        
    
       **Non-Trainable Params** = `2 * 64` = 128
   

3. **Another Hidden Layer with Dropout and Batch Normalization**

    ```python
    model_o.add(Dense(32, activation='relu'))
    model_o.add(BatchNormalization())
    model_o.add(Dropout(0.5))
    ```

    - **Dense Layer**
        - **Input Shape**: `64`
        - **Output Shape**: `32`
        - **Parameters**: `32 * (64 + 1)`
        
        Each neuron has 64 weights plus one bias. Thus, for 32 neurons:
        
        **Params** = `32 * (64 + 1)` = 2080
   

    - **Batch Normalization**
        - **Parameters**: `4 * 32`
        
        Batch Normalization has 4 parameters per neuron: 2 for scale and shift (trainable), and 2 for mean and variance (non-trainable).
        
       
       **Trainable Params** = `2 * 32` = 64
       
        
      
      **Non-Trainable Params**  = `2 * 32` = 64
      

4. **Output Layer with Softmax Activation**

    ```python
    model.add(Dense(len(set(y_train)), activation='softmax'))
    ```

    - **Dense Layer**
        - **Input Shape**: `32`
        - **Output Shape**: `num_classes`
        - **Parameters**: `num_classes * (32 + 1)`
        
        Each class has 32 weights plus one bias. Thus, for `num_classes` classes:
     
        **Params** = `num_classes * (32 + 1)`
      

## Parameters Summary

```plaintext
Total params: 139,360 (544.38 KB)
Trainable params: 138,912 (542.62 KB)
Non-trainable params: 448 (1.75 KB)
