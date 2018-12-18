% This code generates a synthetic sequences consisting of four transition
% models (or sequence constructs)

% Our goal is to infer these TPMs from the sequence

% Author: Harshal Maske  
% created: Jan 2017

% Further code with details will be updated soon
% email: harshalmaske87@gmail.com for ?s and codes
%% Discover set of transition Matrices from a synthetic observation sequence

clear all;close all;clc
seed =5; 
s = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(s);

dim = 3; % number of modes 'K'
s1=[1 2 3 1 2 3 1 2 3];  % Construct for Sequence 1
s2=[1 1 1 2 2 2 3 3 3];  % Construct for Sequence 2
s3=[1 3 2 1 3 2 1 3 2];  % Construct for Sequence 3
s4=[randperm(3) randperm(3) randperm(3)];  % Construct for Sequence 4

O1=[];O2=[];O3=[];O4=[];
% Create long sequences consisting of above construct 
for i=1:5
    O1 = [O1 s1];
    O2 = [O2 s2];
    O3 = [O3 s3];
    O4 = [O4 s4];
end

% Synthetic observation sequence consisting of above sequences
O=[O1 O2 O1 O2 O1 O2 O3 O1 O2 O4 O4 O1 O2 O3 O1];
% ground truth labels
OO = [ones(1,45) 2*ones(1,45) ones(1,45) 2*ones(1,45) ones(1,45) 2*ones(1,45) ...
    3*ones(1,45) ones(1,45) 2*ones(1,45) 4*ones(1,45) 4*ones(1,45) ones(1,45) 2*ones(1,45) 3*ones(1,45) ones(1,45)];

% likelihood rate based estimation (returns inferred TPMs as a structure)
% Tune update condition in line 18 of the following function 
T = deep_markov(O,dim,OO);