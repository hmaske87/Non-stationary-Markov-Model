function [transMat] = deep_markov(O, dim, label)

% inputs and initialization
Mat_count=1;
lkl.val =[];
na = 2;
transMat.count = na.*ones(dim);
transMat.temp_count = na.*ones(dim);

for i=1:dim
        trans_dir(i,:)= normalize(transMat.count(i,:));
        alpha_var(i,:)=dirichlet_var(transMat.count(i,:));
end
transMat.trandir = trans_dir;
transMat.var = alpha_var;

% Knobs and temp variables
n_pi = dim-1; % update condition
var_tol = 1e-4;
size = 3;

for i=1:dim
    dcts(i).count=0; %n_j
end

aind_prev = 0;
U = [];
temp_matcount = zeros(dim);

for k=2:length(O)
    % Mean Variance Estimator
    lambda = 1; 
    I=O(k-1);
    dcts(I).count = dcts(I).count+1;
    for J=1:dim
        if J==O(k)
            prev = trans_dir(I,J);
            trans_dir(I,J) =  trans_dir(I,J) + (1/lambda)*alpha_var(I,J)*(1 - trans_dir(I,J))/(trans_dir(I,J)*(1-trans_dir(I,J)));

            gam = prev*(1-prev)/(trans_dir(I,J)*(1-trans_dir(I,J)));
            alpha_var(I,J) = 1/((1/lambda)*gam*(1/alpha_var(I,J)) + 1/(prev*(1-prev)));
            temp_matcount(I,J)= temp_matcount(I,J)+1;
        else
            prev = trans_dir(I,J);
            trans_dir(I,J) =  trans_dir(I,J) + (1/lambda)*alpha_var(I,J)*(0 - trans_dir(I,J))/(trans_dir(I,J)*(1-trans_dir(I,J)));

            gam = prev*(1-prev)/(trans_dir(I,J)*(1-trans_dir(I,J)));
            alpha_var(I,J) = 1/((1/lambda)*gam*(1/alpha_var(I,J)) + 1/(prev*(1-prev)));
        end
    end
    if abs(sum(trans_dir(I,:))-1)>0.015
        error('row sum is not one');
    end
    
    % Evaluate Likelihood
    for i=1:Mat_count
        DD{i,1} = transMat(i).trandir;
        DD{i,2} = transMat(i).var;
    end
    [new, ~] = dirichlet_lkl_func(trans_dir,DD(1:Mat_count,1),dim);
    for i=1:Mat_count
        if length(lkl)<Mat_count
            lkl(Mat_count).val=[];
            lkl(Mat_count).val(end+k-2)=0;   
        end
        lkl(i).val(end+1)  = new(i);
    end
        no_update = 0;
 % Evaluate Likelihood rate assuming linear model (one could use curvature)
    if k>size
        H = [ones(1,size); k-size+1:k]';
        for i=1:length(new)
            phi(i,:) = vertcat(U,lkl(i).val(k-size:k-1));
        end
        if Mat_count == 1
            lklrate.Z(:,k) = (H'*H)\H'*phi(1,:)';
            lklrate.count = 0;
        else
            phi=normalize(phi,1);
            for i=1:Mat_count
                lklrate(i).Z(:,k) = (H'*H)\H'*phi(i,:)';
                if isempty(lklrate(i).count)
                    lklrate(i).count = 0;
                end
                if lklrate(i).Z(2,k) > 0
                    lklrate(i).count = lklrate(i).count + 1;
                else
                    lklrate(i).count = lklrate(i).count - 1;
                end
            end
        end
        
    if sum(extractfield(dcts, 'count') > n_pi) == dim
        if Mat_count == 1
            transMat.count = transMat.count + temp_matcount;
            for i=1:dim
                trans_dir(i,:)= normalize(transMat.count(i,:));
                alpha_var(i,:)=dirichlet_var(transMat.count(i,:));
            end
            transMat.trandir = trans_dir;
            transMat.var = alpha_var;
            O_a(1:k) = 1;
            transMat.index(1:k) = 1:k;
            k_prev = k+1;
            transMat(2).count = na.*ones(dim);
            transMat(2).temp_count = na.*ones(dim);

            for i=1:dim
                trans_dir(i,:)= normalize(transMat(2).count(i,:));
                alpha_var(i,:)=dirichlet_var(transMat(2).count(i,:));
            end
            transMat(2).trandir = trans_dir;
            transMat(2).var = alpha_var;
            transMat(2).index = [];
            Mat_count = Mat_count +1;
            temp_matcount = zeros(dim);
        else
            [b,aind] = max(extractfield(lklrate, 'count'));
            c =  (extractfield(lklrate, 'count') == b);
           
            if sum(c) ~= 1 
                if lklrate(aind_prev).count == b
                    aind = aind_prev;
                else
                    no_update = 0;
                end
            end
                % train the uniform matrix if it had a highest lklrate
                if aind == Mat_count && no_update == 0
                    transMat(end).count = transMat(end).count + temp_matcount;
                    for i=1:dim
                        trans_dir(i,:)= normalize(transMat(end).count(i,:));
                        alpha_var(i,:)=dirichlet_var(transMat(end).count(i,:));
                    end
                    transMat(end).trandir = trans_dir;
                    transMat(end).var = alpha_var;
                    O_a(end+1:k) = aind;
                    transMat(end).index(end+1:length(k_prev:k)) = k_prev:k;
%                     transMat(end).index = [transMat(aind).index k_prev:k];
                    k_prev = k+1;
                    
                    Mat_count = Mat_count +1;
                    transMat(Mat_count).count = na.*ones(dim);
                    transMat(Mat_count).temp_count = na.*ones(dim);

                    for i=1:dim
                        trans_dir(i,:)= normalize(transMat(Mat_count).count(i,:));
                        alpha_var(i,:)=dirichlet_var(transMat(Mat_count).count(i,:));
                    end
                    transMat(Mat_count).trandir = trans_dir;
                    transMat(Mat_count).var = alpha_var;
                    transMat(Mat_count).index = [];

                elseif no_update == 0 % update the most likely matrix
                    if ~isempty(find((transMat(1).var < var_tol) - ones(dim)))
                        transMat(aind).count = transMat(aind).count + temp_matcount;
                        for i=1:dim
                            trans_dir(i,:)= normalize(transMat(aind).count(i,:));
                            alpha_var(i,:)=dirichlet_var(transMat(aind).count(i,:));
                        end
                        transMat(aind).trandir = trans_dir;
                        transMat(aind).var = alpha_var;  
                    end
                    transMat(aind).index = [transMat(aind).index k_prev:k];
                    k_prev = k+1;
                    O_a(end+1:k) = aind;
                end
                if no_update == 0
                for i=1:Mat_count
                    lklrate(i).count = 0;
                end
            	temp_matcount = zeros(dim);
                aind_prev = aind;
                end
        end
        for i=1:dim
            dcts(i).count=0;
        end
    end
    end
end

% Tackle label switching here


if any(any(transMat(1).count - 2*ones(dim)))
    O_a(end+1:k) = aind_prev;
    transMat(aind_prev).index = [transMat(aind_prev).index k_prev:k];
else
    error('Insufficient Transitions in the given sequence');
end


if nargin > 2
    OO=label;
figure
plot(OO,'k')
hold on
plot(O_a,'r--')
legend('Original','Inferred')
ylim([0 4])
end

