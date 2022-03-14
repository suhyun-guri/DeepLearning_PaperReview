# Deep Residual Learning for Image Recognition

---

## 1. Introduction.

Deep networks는 일반적으로 low/mid/high 레벨의 피쳐들이 적절하게 추출되고 그러한 피쳐들의 레벨 또한 풍부해질 수 있다.

**_의문 :그렇다면 더 많은 층을 쌓으면 더 좋은 networks를 학습시킬 수 있는 것일까?
_**
오래 전부터 단순히 층만 깊게 쌓는 것은 많은 문제를 야기할 수 있다고 알려져 왔다.

vanishing/exploding gradients 문제 → 가중치 값들을 초기에 적절히 초기화하는 것과 중간 정규화 계층에 의해 해결

본 논문은 층이 깊어짐에 따라 degradation 문제가 발생할 수 있다고 주장한다. 즉 층이 깊으면 accuracy가 무조건 높아지는 것이 아니라 어느 정도 이상 깊으면 오히려 accuracy가 감소할 수 있다는 것이다. 또한 이러한 문제는 단순히 overfitting로 야기되는 것이 아니며 층을 깊게 쌓으면 training error가 높아지는 문제가 생길 수 있다.

![](https://images.velog.io/images/suhyun-guri/post/6b656d84-7e7f-4d08-ba81-0f88de745c09/Untitled%20(9).png)

위 그림은 CIFAR-10을 사용한 plain networks의 Training/Test error이다. 더 깊은 층인 56-layer의 error rate가 더 높은 것을 볼 수 있다. 학습 자체가 잘 안되는 것이다.

## 2. Deep ressidual learning.

### Residual Learning

본 논문에서는 degradation 문제를 해결하기 위해 **deep residual learning framework**를 제안한다. (Resnet)

![](https://images.velog.io/images/suhyun-guri/post/f5dc7391-5666-43c8-b76f-da67d30a7b39/Untitled%20(10).png)

F(x)는 weight layer 두 개를 거친 이후의 값을 의미

기존의 기본 매핑을 $H(x)$라고 할 때(x는 layer의 input)이 $H(x)$는 여러 비선형 layer로 이루어져 천천히 복잡한 함수에 근사된다고 가정할 때,  $F(x) := H(x) - x$로 변형시켜 $F(x)+x$를 $H(x)$에 근사하도록 하는 것(Residual mapping)이 더 쉽다고 가정한다. 이를 feed-forward neural network에 적용한 것이 **Shortcut connection**이라고 한다. skip connection이라고도 한다.

> identity mapping : 입력으로 들어간 값 x 가 어떠한 함수를 통과하더라도 다시 x 가 나오는 것

> **일반적인 Network training** <br>
<img src=https://user-images.githubusercontent.com/70987343/158186465-1c0c4e86-2576-45ba-9b55-c4c7bbf2d583.png width=200px> <br>
위 그림은 일반적인 Network training 과정이다.<br>
즉, input $x$가 들어오면 $H(x) = W_2\sigma(W_1X)$라는 $mapping$에서 weight matrix를 훈련 시킨다.<br>
이때 input $x$는 이전 convolution layer의 output이고 weight layer가 convolution layer라고 해보자. input $x$는 convolution의 output이기 때문에 `feature`가 되고 $H(x)$ 역시 `feature`이다. 
여기서 $mapping : H(x)$는 이전에 추출하지 못한 feature들이 추출된다.

> **Deep Residual Learning Framework** <br>
`convolution은 feature을 추출한다.` 라는 컨셉에서 $identity$ $x$는 이전 층에서 학습된 정보이고 $F(x)$는 아직 학습하지 못한 `Residual` 정보이다.<br>
즉, 이미 학습 시킨 $x$를 보존한 상태로 추가적으로 필요한 정보 : $F(x) (=H(x)-x)$만 훈련시켜 학습 난이도를 낮춘다.<br>
- $F(x)$만을 훈련 시킨다고 해서 추가적인 기법이 필요한 것이 아니라 $H(x) = F(x) + x$라고 하게 되면 $F(x)$만 훈련 시키게 된다.


극단적으로 identity mapping이 최적의 해라고 했을 때, 함수 F가 0이 될 수 있게 하는 것이 학습 난이도가 더 쉽다. 다시말해 H가 x인 경우 즉 우리가 본질적으로 학습시키고자 하는 mapping이 identity mapping일 때, residual 자체가 0이 되도록 학습시키는 것이 더 쉽다.
shortcut connection은 단순히 identity mapping으로 사용할 수 있으며 출력값에 단순히 x를 더해주는 것이기 때문에 추가적인 파라미터가 필요하지도 않고 복잡도가 증가하지도 않으며 구현도 간단한 것이 장점이다.

본 논문은 1) residual network를 이용했을 때 학습 난이도가 더 쉽다. 2) residual network는 깊이가 깊어질수록 높은 accuracy를 보인다. 고 말한다.

또한 CIFAR-10, ImageNet에서 모두 성능이 좋아졌으며 특정 데이터셋에 국한된 방법이 아니다라고 한다.

### Identity Mapping by Shortcuts.

하나의 residual block을 다음과 같이 정의한다.

> $y = F(x, \{W_i\}) + x$

$F(x, \{W_i\})$는 residual mapping을 의미하고 $x$는 identity mapping(즉, shortcut connection)을 의미한다.

앞서 Fig 2는 $F = W_2\sigma(W_1x)$ 이렇게 두 개의 weights를 중첩해서 사용하는 것이다. 여기서 $\sigma$는 ReLU이고 biases는 생략되었다.

만약 input($x$) dimension과 ouput($F$) dimension이 다르다고 할 때,

$y = F(x, \{W_i\}) + W_sx$ 이렇게 linear projection인 $W_s$를 곱해줌으로써 dimension을 맞춰준다.

또한 중첩된 layer가 아닌 Single layer의 경우 단순히 Linear layer이므로 ($y = W_1x + x$) 사용시 별다른 이점이 없다고 말하고 있다.

### Plain Network.

비교 목적으로 기본적인 CNN 모델을 가져와서 실험을 진행한다.

본 논문의 Plain network는 VGG Net에서 제안되었던 기법들을 적절히 따르고 있다고 한다.

layer마다 time complexity를 보존할 수 있는 형태로 네트워크를 구성, 별도의 pooling layer를 사용하지 않고 convolution layer에 stride를 2로 줌으로써 downsampling을 진행했다고 한다. 

결과적으로 본 모델은 VGG Net보다 더 적은 파라미터를 사용하고 복잡도 또한 낮았다고 한다. 
<img src=https://images.velog.io/images/suhyun-guri/post/e0f78b31-cb8f-4516-a452-9170c1215282/Screenshot_20220309-164123_Samsung%20Notes.jpg width=400>

<span style="color:grey"> _점선은 input단과 output단의 dimension이 일치하지 않아서 맞춰주기 위한 테크닉이 가미된 부분이다._</span>

VGG와 비교했을 때 FLOPs가 더 감소했다.

> FLOPs : 딥러닝 모델에서 계산 복잡도를 나타내기 위한 척도
> 

### Residual Network.

입력단과 출력단의 dimension이 같을 때 바로 identity mapping을 사용할 수 있다.

입력단과 출력단의 dimension이 다를 때는

1. 사이드에 padding을 해서 identity mapping을 수행
2. projection 연산을 사용해서 구현

### Implementation

실제 구현 상의 테크닉 설명 - ImageNet을 위해 사용

224X224로 랜덤하게 crop, horizontal flip 사용 가능

각 Convolution layer를 거칠 때마다 Batch Normalization 적용

learning rate는 0.1에서 시작해 학습이 진행되면서 점진적으로 줄어들 수 있도록 함

weight decay = 0.0001, momentum = 0.9

## 3. Experiments

### ImageNet 2012 classification dataset

training images 1.28 백만개, validation images는 5만장, test images는 10만장 사용

![](https://images.velog.io/images/suhyun-guri/post/24ff1042-a658-4aec-903c-58247c17602f/Untitled%20(11).png)

Plain network의 경우 layer가 깊을수록 성능이 떨어지고 ResNet의 경우 layer가 깊을수록 성능이 높아지는 것을 확인할 수 있다.

본 연구에서는 이 문제가 vanishing gradients 때문에 발생한 문제가 아니라고 말하고 있다. 이 문제는 수렴률이 기하급수적으로 낮아지는 것이 문제라고 말하고 있다.

> convergence rates : 최적화 기법에서 등장하는 개념으로, 수렴을 위해 필요한 epoch이나 수렴 난이도를 언급하고자 할 때 사용하는 척도이다.
> 

결론적으로 Plain network와 비교했을 때 ResNet의 경우 더 깊은 layer가 얕은 layer에 비해서 잘 동작하고 있고 training rate도 낮고 일반화 성능 또한 높다고 한다.

수렴 속도 또한 더 빠르다는 것을 확인할 수 있다. ResNet 초기 단계에서 더 빠르게 수렴할 수 있도록 만들어줌으로써 optimization 자체를 더욱 쉽게 만들어주는 것이 장점이다.

<img src=https://images.velog.io/images/suhyun-guri/post/2c5278b3-c364-4e56-a1b6-9844df7caf1a/Untitled%20(12).png width=600>

<span style="color:grey"> _오른쪽이 Bottleneck_</span>

또한 본 논문에서 추가적으로 shortcut connection을 위해서 identity mapping과 projection mapping 사용시 결과를 실험을 통해 알려주고 있다.

3가지 방법이 있다.

	A) zero padding으로 dimension 늘려서 사용

	B) dimension이 증가할 때만 projection 연산 수행

	C) 모든 shortcut에 projection 적용

실험 결과, C가 가장 성능이 좋았지만 projection shortcut이 필수적이라고 할 만큼 높은 개선은 아니라고 말하고 있다.

기본적으로 identity shortcut을 이용해서 성능을 많이 개선할 수 있으며 identity shortcut은 파라미터 자체가 없으므로 the bottleneck architectures에 대해서는 파라미터 수를 줄이는 데에 기여할 수 있으며 복잡도를 늘리지 않는 데에 효과적이다.

> bottleneck architectures
  사용 목적 : 연산량을 줄이기 위함, 파라미터 수를 줄이기 위함.
  참고 : [[딥러닝] DeepLearning CNN BottleNeck 원리(Pytorch 구현)](https://coding-yoon.tistory.com/116)
- Standard는 Channel 수가 적을지라도, 3x3 Convolution을 두 번 통과했고,
 BottleNeck은 1x1, 3x3, 1x1 순으로 Convolution을 통과하고, Channel 수는 4배 정도 많지만, Parameter가 세 배 정도 적다.


### CIFAR-10 dataset

32X32 images로 ImageNet보다 훨씬 작다. 그래서 CIFAR-10에 맞춰서 파라미터 수를 줄여서 별도의 ResNet을 고안해서 사용했다고 한다.

즉, ImageNet과 비교했을 때 구조가 다르긴 하지만 유사한 형태를 가지고 있다.

![](https://images.velog.io/images/suhyun-guri/post/1f9b5004-c354-4516-ab37-9350db9124cd/Untitled%20(13).png)

결과를 보면 파라미터 수는 더 적지만 성능이 가장 좋은 것을 볼 수 있다.

### References
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [ResNet: Deep Residual Learning for Image Recognition (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)](https://youtu.be/671BsKl8d0E)
