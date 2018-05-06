clear;
clc;
% -----------加载数据
load('mnist_uint8', 'train_x');
train_x = double(reshape(train_x, 60000, 28, 28))/255;
train_x = permute(train_x,[1,3,2]);
train_x = reshape(train_x, 60000, 784);
% -----------------定义模型
generator = nnsetup([100, 512, 784]);
discriminator = nnsetup([784, 512, 1]);
% -----------开始训练
batch_size = 60;
epoch = 100;
images_num = 60000;
batch_num = ceil(images_num / batch_size);
learning_rate = 0.01;
lamda = 0.001;
for e=1:epoch
    kk = randperm(images_num);
    for t=1:batch_num
        % 准备数据
        images_real = train_x(kk((t - 1) * batch_size + 1:t * batch_size), :, :);
        noise = unifrnd(-1, 1, batch_size, 100);
        % 开始训练
        % -----------更新generator，固定discriminator
        generator = nnff(generator, noise);
        images_fake = generator.layers{generator.layers_count}.a;
        discriminator = nnff(discriminator, images_fake);
        logits_fake = discriminator.layers{discriminator.layers_count}.z;
        discriminator = nnbp_d(discriminator, logits_fake, ones(batch_size, 1));
        generator = nnbp_g(generator, discriminator);
        generator = nnapplygrade(generator, learning_rate, lamda);
        % -----------更新discriminator，固定generator
        generator = nnff(generator, noise);
        images_fake = generator.layers{generator.layers_count}.a;
        images = [images_fake;images_real];
        discriminator = nnff(discriminator, images);
        logits = discriminator.layers{discriminator.layers_count}.z;
        labels = [zeros(batch_size,1);ones(batch_size,1)];
        discriminator = nnbp_d(discriminator, logits, labels);
        discriminator = nnapplygrade(discriminator, learning_rate, lamda);
        % ----------------输出loss
        if t == batch_num
            c_loss = sigmoid_cross_entropy(logits(1:batch_size), ones(batch_size, 1));
            d_loss = sigmoid_cross_entropy(logits, labels);
            fprintf('c_loss:"%f",d_loss:"%f"\n',c_loss, d_loss);
        end
        if t == batch_num
            path = ['./pics/epoch_',int2str(e),'_t_',int2str(t),'.png'];
            save_images(images_fake, [4, 4], path);
            fprintf('save_sample:%s\n', path);
        end
    end
end
% sigmoid激活函数
function output = sigmoid(x)
    output =1./(1+exp(-x));
end
% relu
function output = relu(x)
    output = max(x, 0);
end
% relu对x的导数
function output = detal_relu(x)
    output = max(x,0);
    output(output>0) = 1;
end
% 交叉熵损失函数，此处的logits是未经过sigmoid激活的
% https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
function result = sigmoid_cross_entropy(logits, labels)
    result = max(logits, 0) - logits .* labels + log(1 + exp(-abs(logits)));
    result = mean(result);
end
% sigmoid_cross_entropy对logits的导数，此处的logits是未经过sigmoid激活的
function result = detal_sigmoid_cross_entropy(logits, labels)
    temp1 = max(logits, 0);
    temp1(temp1>0) = 1;
    temp2 = logits;
    temp2(temp2>0) = -1;
    temp2(temp2<0) = 1;
    result = temp1 - labels + exp(-abs(logits))./(1+exp(-abs(logits))) .* temp2;
end
% 根据所给的结构建立网络
function nn = nnsetup(architecture)
    nn.architecture   = architecture;
    nn.layers_count = numel(nn.architecture);
    % 假设结构为[100, 512, 784]，则有3层，输入层100，两个隐藏层：100*512，512*784, 输出为最后一层的a值（激活值）
    for i = 2 : nn.layers_count   
        nn.layers{i}.w = normrnd(0, 0.02, nn.architecture(i-1), nn.architecture(i));
        nn.layers{i}.b = normrnd(0, 0.02, 1, nn.architecture(i));
    end
end
% 前向传递
function nn = nnff(nn, x)
    nn.layers{1}.a = x;
    for i = 2 : nn.layers_count
        input = nn.layers{i-1}.a;
        w = nn.layers{i}.w;
        b = nn.layers{i}.b;
        nn.layers{i}.z = input*w + repmat(b, size(input, 1), 1);
        if i ~= nn.layers_count
            nn.layers{i}.a = relu(nn.layers{i}.z);
        else
            nn.layers{i}.a = sigmoid(nn.layers{i}.z);
        end
    end
end
% discriminator的bp，下面的bp涉及到对各个参数的求导
% 如果更改网络结构（激活函数等）则涉及到bp的更改，更改weights，biases的个数则不需要更改bp
% 为了更新w,b，就是要求最终的loss对w，b的偏导数，残差就是在求w，b偏导数的中间计算过程的结果
function nn = nnbp_d(nn, y_h, y)
    % d表示残差，残差就是最终的loss对各层未激活值（z）的偏导，偏导数的计算需要采用链式求导法则-自己手动推出来
    n = nn.layers_count;
    % 最后一层的残差
    nn.layers{n}.d = detal_sigmoid_cross_entropy(y_h, y);
    for i = n-1:-1:2
        d = nn.layers{i+1}.d;
        w = nn.layers{i+1}.w;
        z = nn.layers{i}.z;
        % 每一层的残差是对每一层的未激活值求偏导数，所以是后一层的残差乘上w,再乘上对激活值对未激活值的偏导数
        nn.layers{i}.d = d*w' .* detal_relu(z);    
    end
    % 求出各层的残差之后，就可以根据残差求出最终loss对weights和biases的偏导数
    for i = 2:n
        d = nn.layers{i}.d;
        a = nn.layers{i-1}.a;
        % dw是对每层的weights进行偏导数的求解
        nn.layers{i}.dw = a'*d / size(d, 1);
        nn.layers{i}.db = mean(d, 1);
    end
end
% generator的bp
function g_net = nnbp_g(g_net, d_net)
    n = g_net.layers_count;
    a = g_net.layers{n}.a;
    % generator的loss是由label_fake得到的，(images_fake过discriminator得到label_fake)
    % 对g进行bp的时候，可以将g和d看成是一个整体
    % g最后一层的残差等于d第2层的残差乘上(a .* (a_o))
    g_net.layers{n}.d = d_net.layers{2}.d * d_net.layers{2}.w' .* (a .* (1-a));
    for i = n-1:-1:2
        d = g_net.layers{i+1}.d;
        w = g_net.layers{i+1}.w;
        z = g_net.layers{i}.z;
        % 每一层的残差是对每一层的未激活值求偏导数，所以是后一层的残差乘上w,再乘上对激活值对未激活值的偏导数
        g_net.layers{i}.d = d*w' .* detal_relu(z);    
    end
    % 求出各层的残差之后，就可以根据残差求出最终loss对weights和biases的偏导数
    for i = 2:n
        d = g_net.layers{i}.d;
        a = g_net.layers{i-1}.a;
        % dw是对每层的weights进行偏导数的求解
        g_net.layers{i}.dw = a'*d / size(d, 1);
        g_net.layers{i}.db = mean(d, 1);
    end
end
% 应用梯度
function nn = nnapplygrade(nn, learning_rate, lamda)
    n = nn.layers_count;
    for i = 2:n
        dw = nn.layers{i}.dw;
        db = nn.layers{i}.db;
        nn.layers{i}.w = nn.layers{i}.w - learning_rate * (dw + lamda * nn.layers{i}.w);
        nn.layers{i}.b = nn.layers{i}.b - learning_rate * db;
    end
end
% 保存图片，便于观察generator生成的images_fake
function save_images(images, count, path)
    n = size(images, 1);
    row = count(1);
    col = count(2);
    I = zeros(row*28, col*28);
    for i = 1:row
        for j = 1:col
            r_s = (i-1)*28+1;
            c_s = (j-1)*28+1;
            index = (i-1)*col + j;
            pic = reshape(images(index, :), 28, 28);
            I(r_s:r_s+27, c_s:c_s+27) = pic;
        end
    end
    imwrite(I, path);
end