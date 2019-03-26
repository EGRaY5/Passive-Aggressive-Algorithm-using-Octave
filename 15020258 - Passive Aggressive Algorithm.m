%Loading the data from CSV file 
data = csvread('datafile.csv');

%Removing the first attribute
data(:,1) = [];

%Replacing the 11th attribute which is now 10 for Benign -1 and Malignant +1
%Take values which equal to 2 in 10th column and then take that data in the 10th column and equal it to -1
data(data(:,10)==2,10)= -1;
data(data(:,10)==4,10)= 1;


%Input Matrix (As first attribute removed values changes)
%For x the first 9 columns are assigned and for Y the 10th column is assigned.
X = data(:,[1:9]);
X(:,10)=1;
X
Y = data(:,10);

%2/3 of seperated Training data
X_Training = X([1:466],:);
Y_Training = Y([1:466],:);

%1/3 of seperated Testing data 
X_Testing = X([467:699],:);
Y_Testing = Y([467:699],:); 

%prediction output file
file_id = fopen('prediction.txt', 'w');
fdisp(file_id, "Outcome of Predictions");

%w1 and accuracy are outputs and iterations,X_Training,Y_Training,algo and file_id are inputs.
function [w1,accuracy] =  training(iteration,X_Training,Y_Training,algo,file_id)
  %w1 assigned to first 9 columns
  w1 = [0,0,0,0,0,0,0,0,0,0];
  %iteraation value taken from below defined Iterations array and the function which is called below it.(1,2,10)
  for i = 1:iteration
    % For first 466 rows
    for j = 1:466
      %Training with C=1 as given in question
      C = 1;
      x = X_Training(j,:);
      y = Y_Training(j,:);
      lt = max(0,1-y*(w1*x'));
      
 %set values according to PA,PA-I,PA-II in research paper
      switch (algo)
          case 1
            Tt = lt/(norm(x'))**2;
          case 2
            Tt = min(C,lt/(norm(x'))**2);
          case 3
            Tt = lt/((norm(x'))**2+ 1/(2*C) );
          otherwise
            error('Algorithm is invalid');
        endswitch
      
      w1 = w1+Tt*(y.*x);
    endfor
  endfor 
  

%Code to test accuracy of the training set
  incacc= 0;
  for j = 1:466  
    x = X_Training(j,:);
    y = Y_Training(j,:);
    %Get the prediction value by multiplying the weight vector and the training data set
    y_prediction = sign(w1*x');
    if(y_prediction*y == 1)
      incacc++;
    endif
  endfor
  accuracy = incacc/466*100;
endfunction

%Code to test accuracy of the Testing data set
function [accuracy] = testing(w1,X_Testing,Y_Testing,file_id)
  incacc = 0;
  for i = 1:233
    x = X_Testing(i,:);
    y = Y_Testing(i,:);
    y_prediction = sign(w1*x');
    if(y_prediction*y == 1)
      fprintf(file_id,"%d is Correct\n",(466+i));
      incacc++;
    else
      fprintf(file_id,"%d is Incorrect\n",(466+i));
    endif
  endfor
  accuracy = incacc/233*100;
endfunction

%Assigning cases for algorithms as PA = 1,PA-I = 2,PA-II = 3 

for Algos = [1,2,3]
  switch(Algos)
    case 1
      Algorithm = "PA";
    case 2
      Algorithm = "PA - I";
    case 3
      Algorithm = "PA - II";
   endswitch
   
%Showing accuracy of training and testing sets according to alogorithms and iterations 1,2,10
  for Iterations = [1,2,10] 
    
%Print line in prediction.txt for seperation
    fprintf(file_id,"-------------------------------------------------------------------------------------------------------------------------------------------\n");

%Printing output in command window
    Iterations
    Algorithm
    [Weights, Training_accuracy] = training(Iterations,X_Training,Y_Training,Algos,file_id);
    Weights 
    [Testing_accuracy] = testing(Weights,X_Testing,Y_Testing,file_id);
    Training_accuracy
    Testing_accuracy
    printf("\n")
    
%Printing result of iteration in prediction.txt after each iteration with relevant spacing
    fprintf(file_id,"\n");
    fprintf(file_id,"For Iterations = %d and Algorithm = %s => Training Accuracy = %f and Testing Accuracy = %f\n",Iterations,Algorithm,Training_accuracy,Testing_accuracy );
    fprintf(file_id,"\n");
    
  endfor
endfor

fclose(file_id);

  



