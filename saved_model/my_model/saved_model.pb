??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8Û
?
conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_37/kernel
}
$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*&
_output_shapes
: *
dtype0
t
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_37/bias
m
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes
: *
dtype0
?
conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*!
shared_nameconv2d_38/kernel
}
$conv2d_38/kernel/Read/ReadVariableOpReadVariableOpconv2d_38/kernel*&
_output_shapes
: 0*
dtype0
t
conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_38/bias
m
"conv2d_38/bias/Read/ReadVariableOpReadVariableOpconv2d_38/bias*
_output_shapes
:0*
dtype0
|
dense_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@* 
shared_namedense_55/kernel
u
#dense_55/kernel/Read/ReadVariableOpReadVariableOpdense_55/kernel* 
_output_shapes
:
??@*
dtype0
r
dense_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_55/bias
k
!dense_55/bias/Read/ReadVariableOpReadVariableOpdense_55/bias*
_output_shapes
:@*
dtype0
z
dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_56/kernel
s
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel*
_output_shapes

:@@*
dtype0
r
dense_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_56/bias
k
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes
:@*
dtype0
z
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_57/kernel
s
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes

:@@*
dtype0
r
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_57/bias
k
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes
:@*
dtype0
z
dense_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_58/kernel
s
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel*
_output_shapes

:@*
dtype0
r
dense_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_58/bias
k
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
?
Adam/conv2d_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_37/kernel/m
?
+Adam/conv2d_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_37/bias/m
{
)Adam/conv2d_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*(
shared_nameAdam/conv2d_38/kernel/m
?
+Adam/conv2d_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/m*&
_output_shapes
: 0*
dtype0
?
Adam/conv2d_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_nameAdam/conv2d_38/bias/m
{
)Adam/conv2d_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/m*
_output_shapes
:0*
dtype0
?
Adam/dense_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*'
shared_nameAdam/dense_55/kernel/m
?
*Adam/dense_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/m* 
_output_shapes
:
??@*
dtype0
?
Adam/dense_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_55/bias/m
y
(Adam/dense_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_56/kernel/m
?
*Adam/dense_56/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_56/bias/m
y
(Adam/dense_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_57/kernel/m
?
*Adam/dense_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_57/bias/m
y
(Adam/dense_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_58/kernel/m
?
*Adam/dense_58/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_58/bias/m
y
(Adam/dense_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_37/kernel/v
?
+Adam/conv2d_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_37/bias/v
{
)Adam/conv2d_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*(
shared_nameAdam/conv2d_38/kernel/v
?
+Adam/conv2d_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/v*&
_output_shapes
: 0*
dtype0
?
Adam/conv2d_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_nameAdam/conv2d_38/bias/v
{
)Adam/conv2d_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/v*
_output_shapes
:0*
dtype0
?
Adam/dense_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*'
shared_nameAdam/dense_55/kernel/v
?
*Adam/dense_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/v* 
_output_shapes
:
??@*
dtype0
?
Adam/dense_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_55/bias/v
y
(Adam/dense_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_56/kernel/v
?
*Adam/dense_56/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_56/bias/v
y
(Adam/dense_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_57/kernel/v
?
*Adam/dense_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_57/bias/v
y
(Adam/dense_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_58/kernel/v
?
*Adam/dense_58/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_58/bias/v
y
(Adam/dense_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?P
value?PB?P B?P
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
R
&	variables
'regularization_losses
(trainable_variables
)	keras_api
R
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
h

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
R
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
?
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratem?m? m?!m?.m?/m?4m?5m?:m?;m?Dm?Em?v?v? v?!v?.v?/v?4v?5v?:v?;v?Dv?Ev?
V
0
1
 2
!3
.4
/5
46
57
:8
;9
D10
E11
 
V
0
1
 2
!3
.4
/5
46
57
:8
;9
D10
E11
?
Olayer_regularization_losses
	variables
regularization_losses
trainable_variables

Players
Qnon_trainable_variables
Rmetrics
Slayer_metrics
 
\Z
VARIABLE_VALUEconv2d_37/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_37/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Tlayer_regularization_losses
	variables
regularization_losses
trainable_variables

Ulayers
Vnon_trainable_variables
Wmetrics
Xlayer_metrics
 
 
 
?
Ylayer_regularization_losses
	variables
regularization_losses
trainable_variables

Zlayers
[non_trainable_variables
\metrics
]layer_metrics
 
 
 
?
^layer_regularization_losses
	variables
regularization_losses
trainable_variables

_layers
`non_trainable_variables
ametrics
blayer_metrics
\Z
VARIABLE_VALUEconv2d_38/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_38/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
?
clayer_regularization_losses
"	variables
#regularization_losses
$trainable_variables

dlayers
enon_trainable_variables
fmetrics
glayer_metrics
 
 
 
?
hlayer_regularization_losses
&	variables
'regularization_losses
(trainable_variables

ilayers
jnon_trainable_variables
kmetrics
llayer_metrics
 
 
 
?
mlayer_regularization_losses
*	variables
+regularization_losses
,trainable_variables

nlayers
onon_trainable_variables
pmetrics
qlayer_metrics
[Y
VARIABLE_VALUEdense_55/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_55/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
?
rlayer_regularization_losses
0	variables
1regularization_losses
2trainable_variables

slayers
tnon_trainable_variables
umetrics
vlayer_metrics
[Y
VARIABLE_VALUEdense_56/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_56/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
?
wlayer_regularization_losses
6	variables
7regularization_losses
8trainable_variables

xlayers
ynon_trainable_variables
zmetrics
{layer_metrics
[Y
VARIABLE_VALUEdense_57/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_57/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
?
|layer_regularization_losses
<	variables
=regularization_losses
>trainable_variables

}layers
~non_trainable_variables
metrics
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
@	variables
Aregularization_losses
Btrainable_variables
?layers
?non_trainable_variables
?metrics
?layer_metrics
[Y
VARIABLE_VALUEdense_58/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_58/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
?
 ?layer_regularization_losses
F	variables
Gregularization_losses
Htrainable_variables
?layers
?non_trainable_variables
?metrics
?layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
?0
?1
?2
?3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
\
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api
\
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv2d_37/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_37/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_38/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_38/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_55/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_55/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_56/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_56/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_57/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_57/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_58/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_58/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_37/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_37/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_38/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_38/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_55/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_55/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_56/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_56/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_57/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_57/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_58/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_58/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_37_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_37_inputconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasdense_55/kerneldense_55/biasdense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_82841
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOp$conv2d_38/kernel/Read/ReadVariableOp"conv2d_38/bias/Read/ReadVariableOp#dense_55/kernel/Read/ReadVariableOp!dense_55/bias/Read/ReadVariableOp#dense_56/kernel/Read/ReadVariableOp!dense_56/bias/Read/ReadVariableOp#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOp#dense_58/kernel/Read/ReadVariableOp!dense_58/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp+Adam/conv2d_37/kernel/m/Read/ReadVariableOp)Adam/conv2d_37/bias/m/Read/ReadVariableOp+Adam/conv2d_38/kernel/m/Read/ReadVariableOp)Adam/conv2d_38/bias/m/Read/ReadVariableOp*Adam/dense_55/kernel/m/Read/ReadVariableOp(Adam/dense_55/bias/m/Read/ReadVariableOp*Adam/dense_56/kernel/m/Read/ReadVariableOp(Adam/dense_56/bias/m/Read/ReadVariableOp*Adam/dense_57/kernel/m/Read/ReadVariableOp(Adam/dense_57/bias/m/Read/ReadVariableOp*Adam/dense_58/kernel/m/Read/ReadVariableOp(Adam/dense_58/bias/m/Read/ReadVariableOp+Adam/conv2d_37/kernel/v/Read/ReadVariableOp)Adam/conv2d_37/bias/v/Read/ReadVariableOp+Adam/conv2d_38/kernel/v/Read/ReadVariableOp)Adam/conv2d_38/bias/v/Read/ReadVariableOp*Adam/dense_55/kernel/v/Read/ReadVariableOp(Adam/dense_55/bias/v/Read/ReadVariableOp*Adam/dense_56/kernel/v/Read/ReadVariableOp(Adam/dense_56/bias/v/Read/ReadVariableOp*Adam/dense_57/kernel/v/Read/ReadVariableOp(Adam/dense_57/bias/v/Read/ReadVariableOp*Adam/dense_58/kernel/v/Read/ReadVariableOp(Adam/dense_58/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_83477
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_37/kernelconv2d_37/biasconv2d_38/kernelconv2d_38/biasdense_55/kerneldense_55/biasdense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1true_positivesfalse_positivestrue_positives_1false_negativesAdam/conv2d_37/kernel/mAdam/conv2d_37/bias/mAdam/conv2d_38/kernel/mAdam/conv2d_38/bias/mAdam/dense_55/kernel/mAdam/dense_55/bias/mAdam/dense_56/kernel/mAdam/dense_56/bias/mAdam/dense_57/kernel/mAdam/dense_57/bias/mAdam/dense_58/kernel/mAdam/dense_58/bias/mAdam/conv2d_37/kernel/vAdam/conv2d_37/bias/vAdam/conv2d_38/kernel/vAdam/conv2d_38/bias/vAdam/dense_55/kernel/vAdam/dense_55/bias/vAdam/dense_56/kernel/vAdam/dense_56/bias/vAdam/dense_57/kernel/vAdam/dense_57/bias/vAdam/dense_58/kernel/vAdam/dense_58/bias/v*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_83634??

?V
?

 __inference__wrapped_model_82251
conv2d_37_input:
6sequential_20_conv2d_37_conv2d_readvariableop_resource;
7sequential_20_conv2d_37_biasadd_readvariableop_resource:
6sequential_20_conv2d_38_conv2d_readvariableop_resource;
7sequential_20_conv2d_38_biasadd_readvariableop_resource9
5sequential_20_dense_55_matmul_readvariableop_resource:
6sequential_20_dense_55_biasadd_readvariableop_resource9
5sequential_20_dense_56_matmul_readvariableop_resource:
6sequential_20_dense_56_biasadd_readvariableop_resource9
5sequential_20_dense_57_matmul_readvariableop_resource:
6sequential_20_dense_57_biasadd_readvariableop_resource9
5sequential_20_dense_58_matmul_readvariableop_resource:
6sequential_20_dense_58_biasadd_readvariableop_resource
identity??.sequential_20/conv2d_37/BiasAdd/ReadVariableOp?-sequential_20/conv2d_37/Conv2D/ReadVariableOp?.sequential_20/conv2d_38/BiasAdd/ReadVariableOp?-sequential_20/conv2d_38/Conv2D/ReadVariableOp?-sequential_20/dense_55/BiasAdd/ReadVariableOp?,sequential_20/dense_55/MatMul/ReadVariableOp?-sequential_20/dense_56/BiasAdd/ReadVariableOp?,sequential_20/dense_56/MatMul/ReadVariableOp?-sequential_20/dense_57/BiasAdd/ReadVariableOp?,sequential_20/dense_57/MatMul/ReadVariableOp?-sequential_20/dense_58/BiasAdd/ReadVariableOp?,sequential_20/dense_58/MatMul/ReadVariableOp?
-sequential_20/conv2d_37/Conv2D/ReadVariableOpReadVariableOp6sequential_20_conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_20/conv2d_37/Conv2D/ReadVariableOp?
sequential_20/conv2d_37/Conv2DConv2Dconv2d_37_input5sequential_20/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2 
sequential_20/conv2d_37/Conv2D?
.sequential_20/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp7sequential_20_conv2d_37_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_20/conv2d_37/BiasAdd/ReadVariableOp?
sequential_20/conv2d_37/BiasAddBiasAdd'sequential_20/conv2d_37/Conv2D:output:06sequential_20/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2!
sequential_20/conv2d_37/BiasAdd?
sequential_20/conv2d_37/ReluRelu(sequential_20/conv2d_37/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
sequential_20/conv2d_37/Relu?
&sequential_20/max_pooling2d_32/MaxPoolMaxPool*sequential_20/conv2d_37/Relu:activations:0*/
_output_shapes
:?????????UU *
ksize
*
paddingVALID*
strides
2(
&sequential_20/max_pooling2d_32/MaxPool?
!sequential_20/dropout_30/IdentityIdentity/sequential_20/max_pooling2d_32/MaxPool:output:0*
T0*/
_output_shapes
:?????????UU 2#
!sequential_20/dropout_30/Identity?
-sequential_20/conv2d_38/Conv2D/ReadVariableOpReadVariableOp6sequential_20_conv2d_38_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02/
-sequential_20/conv2d_38/Conv2D/ReadVariableOp?
sequential_20/conv2d_38/Conv2DConv2D*sequential_20/dropout_30/Identity:output:05sequential_20/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????SS0*
paddingVALID*
strides
2 
sequential_20/conv2d_38/Conv2D?
.sequential_20/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp7sequential_20_conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype020
.sequential_20/conv2d_38/BiasAdd/ReadVariableOp?
sequential_20/conv2d_38/BiasAddBiasAdd'sequential_20/conv2d_38/Conv2D:output:06sequential_20/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????SS02!
sequential_20/conv2d_38/BiasAdd?
sequential_20/conv2d_38/ReluRelu(sequential_20/conv2d_38/BiasAdd:output:0*
T0*/
_output_shapes
:?????????SS02
sequential_20/conv2d_38/Relu?
&sequential_20/max_pooling2d_33/MaxPoolMaxPool*sequential_20/conv2d_38/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2(
&sequential_20/max_pooling2d_33/MaxPool?
sequential_20/flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2 
sequential_20/flatten_18/Const?
 sequential_20/flatten_18/ReshapeReshape/sequential_20/max_pooling2d_33/MaxPool:output:0'sequential_20/flatten_18/Const:output:0*
T0*)
_output_shapes
:???????????2"
 sequential_20/flatten_18/Reshape?
,sequential_20/dense_55/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_55_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02.
,sequential_20/dense_55/MatMul/ReadVariableOp?
sequential_20/dense_55/MatMulMatMul)sequential_20/flatten_18/Reshape:output:04sequential_20/dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_20/dense_55/MatMul?
-sequential_20/dense_55/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_20/dense_55/BiasAdd/ReadVariableOp?
sequential_20/dense_55/BiasAddBiasAdd'sequential_20/dense_55/MatMul:product:05sequential_20/dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2 
sequential_20/dense_55/BiasAdd?
sequential_20/dense_55/ReluRelu'sequential_20/dense_55/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential_20/dense_55/Relu?
,sequential_20/dense_56/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_56_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02.
,sequential_20/dense_56/MatMul/ReadVariableOp?
sequential_20/dense_56/MatMulMatMul)sequential_20/dense_55/Relu:activations:04sequential_20/dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_20/dense_56/MatMul?
-sequential_20/dense_56/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_56_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_20/dense_56/BiasAdd/ReadVariableOp?
sequential_20/dense_56/BiasAddBiasAdd'sequential_20/dense_56/MatMul:product:05sequential_20/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2 
sequential_20/dense_56/BiasAdd?
sequential_20/dense_56/ReluRelu'sequential_20/dense_56/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential_20/dense_56/Relu?
,sequential_20/dense_57/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_57_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02.
,sequential_20/dense_57/MatMul/ReadVariableOp?
sequential_20/dense_57/MatMulMatMul)sequential_20/dense_56/Relu:activations:04sequential_20/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_20/dense_57/MatMul?
-sequential_20/dense_57/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_57_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_20/dense_57/BiasAdd/ReadVariableOp?
sequential_20/dense_57/BiasAddBiasAdd'sequential_20/dense_57/MatMul:product:05sequential_20/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2 
sequential_20/dense_57/BiasAdd?
sequential_20/dense_57/ReluRelu'sequential_20/dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential_20/dense_57/Relu?
!sequential_20/dropout_31/IdentityIdentity)sequential_20/dense_57/Relu:activations:0*
T0*'
_output_shapes
:?????????@2#
!sequential_20/dropout_31/Identity?
,sequential_20/dense_58/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_58_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,sequential_20/dense_58/MatMul/ReadVariableOp?
sequential_20/dense_58/MatMulMatMul*sequential_20/dropout_31/Identity:output:04sequential_20/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_20/dense_58/MatMul?
-sequential_20/dense_58/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_20/dense_58/BiasAdd/ReadVariableOp?
sequential_20/dense_58/BiasAddBiasAdd'sequential_20/dense_58/MatMul:product:05sequential_20/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_20/dense_58/BiasAdd?
sequential_20/dense_58/SigmoidSigmoid'sequential_20/dense_58/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_20/dense_58/Sigmoid?
IdentityIdentity"sequential_20/dense_58/Sigmoid:y:0/^sequential_20/conv2d_37/BiasAdd/ReadVariableOp.^sequential_20/conv2d_37/Conv2D/ReadVariableOp/^sequential_20/conv2d_38/BiasAdd/ReadVariableOp.^sequential_20/conv2d_38/Conv2D/ReadVariableOp.^sequential_20/dense_55/BiasAdd/ReadVariableOp-^sequential_20/dense_55/MatMul/ReadVariableOp.^sequential_20/dense_56/BiasAdd/ReadVariableOp-^sequential_20/dense_56/MatMul/ReadVariableOp.^sequential_20/dense_57/BiasAdd/ReadVariableOp-^sequential_20/dense_57/MatMul/ReadVariableOp.^sequential_20/dense_58/BiasAdd/ReadVariableOp-^sequential_20/dense_58/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2`
.sequential_20/conv2d_37/BiasAdd/ReadVariableOp.sequential_20/conv2d_37/BiasAdd/ReadVariableOp2^
-sequential_20/conv2d_37/Conv2D/ReadVariableOp-sequential_20/conv2d_37/Conv2D/ReadVariableOp2`
.sequential_20/conv2d_38/BiasAdd/ReadVariableOp.sequential_20/conv2d_38/BiasAdd/ReadVariableOp2^
-sequential_20/conv2d_38/Conv2D/ReadVariableOp-sequential_20/conv2d_38/Conv2D/ReadVariableOp2^
-sequential_20/dense_55/BiasAdd/ReadVariableOp-sequential_20/dense_55/BiasAdd/ReadVariableOp2\
,sequential_20/dense_55/MatMul/ReadVariableOp,sequential_20/dense_55/MatMul/ReadVariableOp2^
-sequential_20/dense_56/BiasAdd/ReadVariableOp-sequential_20/dense_56/BiasAdd/ReadVariableOp2\
,sequential_20/dense_56/MatMul/ReadVariableOp,sequential_20/dense_56/MatMul/ReadVariableOp2^
-sequential_20/dense_57/BiasAdd/ReadVariableOp-sequential_20/dense_57/BiasAdd/ReadVariableOp2\
,sequential_20/dense_57/MatMul/ReadVariableOp,sequential_20/dense_57/MatMul/ReadVariableOp2^
-sequential_20/dense_58/BiasAdd/ReadVariableOp-sequential_20/dense_58/BiasAdd/ReadVariableOp2\
,sequential_20/dense_58/MatMul/ReadVariableOp,sequential_20/dense_58/MatMul/ReadVariableOp:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_37_input
?	
?
C__inference_dense_58_layer_call_and_return_conditional_losses_82519

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
E__inference_flatten_18_layer_call_and_return_conditional_losses_83150

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
~
)__inference_conv2d_37_layer_call_fn_83085

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_822962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_18_layer_call_fn_83155

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_18_layer_call_and_return_conditional_losses_823832
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
~
)__inference_conv2d_38_layer_call_fn_83144

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????SS0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_823602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????SS02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????UU ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????UU 
 
_user_specified_nameinputs
?
c
E__inference_dropout_30_layer_call_and_return_conditional_losses_83102

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????UU 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????UU 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????UU :W S
/
_output_shapes
:?????????UU 
 
_user_specified_nameinputs
?
a
E__inference_flatten_18_layer_call_and_return_conditional_losses_82383

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
D__inference_conv2d_37_layer_call_and_return_conditional_losses_83076

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
2conv2d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_37/kernel/Regularizer/SquareSquare:conv2d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_37/kernel/Regularizer/Square?
"conv2d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_37/kernel/Regularizer/Const?
 conv2d_37/kernel/Regularizer/SumSum'conv2d_37/kernel/Regularizer/Square:y:0+conv2d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/Sum?
"conv2d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_37/kernel/Regularizer/mul/x?
 conv2d_37/kernel/Regularizer/mulMul+conv2d_37/kernel/Regularizer/mul/x:output:0)conv2d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_37/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2conv2d_37/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?t
?	
H__inference_sequential_20_layer_call_and_return_conditional_losses_82925

inputs,
(conv2d_37_conv2d_readvariableop_resource-
)conv2d_37_biasadd_readvariableop_resource,
(conv2d_38_conv2d_readvariableop_resource-
)conv2d_38_biasadd_readvariableop_resource+
'dense_55_matmul_readvariableop_resource,
(dense_55_biasadd_readvariableop_resource+
'dense_56_matmul_readvariableop_resource,
(dense_56_biasadd_readvariableop_resource+
'dense_57_matmul_readvariableop_resource,
(dense_57_biasadd_readvariableop_resource+
'dense_58_matmul_readvariableop_resource,
(dense_58_biasadd_readvariableop_resource
identity?? conv2d_37/BiasAdd/ReadVariableOp?conv2d_37/Conv2D/ReadVariableOp?2conv2d_37/kernel/Regularizer/Square/ReadVariableOp? conv2d_38/BiasAdd/ReadVariableOp?conv2d_38/Conv2D/ReadVariableOp?2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?dense_55/BiasAdd/ReadVariableOp?dense_55/MatMul/ReadVariableOp?1dense_55/kernel/Regularizer/Square/ReadVariableOp?dense_56/BiasAdd/ReadVariableOp?dense_56/MatMul/ReadVariableOp?dense_57/BiasAdd/ReadVariableOp?dense_57/MatMul/ReadVariableOp?dense_58/BiasAdd/ReadVariableOp?dense_58/MatMul/ReadVariableOp?
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_37/Conv2D/ReadVariableOp?
conv2d_37/Conv2DConv2Dinputs'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
conv2d_37/Conv2D?
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp?
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_37/BiasAdd?
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_37/Relu?
max_pooling2d_32/MaxPoolMaxPoolconv2d_37/Relu:activations:0*/
_output_shapes
:?????????UU *
ksize
*
paddingVALID*
strides
2
max_pooling2d_32/MaxPooly
dropout_30/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_30/dropout/Const?
dropout_30/dropout/MulMul!max_pooling2d_32/MaxPool:output:0!dropout_30/dropout/Const:output:0*
T0*/
_output_shapes
:?????????UU 2
dropout_30/dropout/Mul?
dropout_30/dropout/ShapeShape!max_pooling2d_32/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_30/dropout/Shape?
/dropout_30/dropout/random_uniform/RandomUniformRandomUniform!dropout_30/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????UU *
dtype021
/dropout_30/dropout/random_uniform/RandomUniform?
!dropout_30/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_30/dropout/GreaterEqual/y?
dropout_30/dropout/GreaterEqualGreaterEqual8dropout_30/dropout/random_uniform/RandomUniform:output:0*dropout_30/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????UU 2!
dropout_30/dropout/GreaterEqual?
dropout_30/dropout/CastCast#dropout_30/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????UU 2
dropout_30/dropout/Cast?
dropout_30/dropout/Mul_1Muldropout_30/dropout/Mul:z:0dropout_30/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????UU 2
dropout_30/dropout/Mul_1?
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02!
conv2d_38/Conv2D/ReadVariableOp?
conv2d_38/Conv2DConv2Ddropout_30/dropout/Mul_1:z:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????SS0*
paddingVALID*
strides
2
conv2d_38/Conv2D?
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp?
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????SS02
conv2d_38/BiasAdd~
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*/
_output_shapes
:?????????SS02
conv2d_38/Relu?
max_pooling2d_33/MaxPoolMaxPoolconv2d_38/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_33/MaxPoolu
flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
flatten_18/Const?
flatten_18/ReshapeReshape!max_pooling2d_33/MaxPool:output:0flatten_18/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_18/Reshape?
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02 
dense_55/MatMul/ReadVariableOp?
dense_55/MatMulMatMulflatten_18/Reshape:output:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_55/MatMul?
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_55/BiasAdd/ReadVariableOp?
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_55/BiasAdds
dense_55/ReluReludense_55/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_55/Relu?
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_56/MatMul/ReadVariableOp?
dense_56/MatMulMatMuldense_55/Relu:activations:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_56/MatMul?
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_56/BiasAdd/ReadVariableOp?
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_56/BiasAdds
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_56/Relu?
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_57/MatMul/ReadVariableOp?
dense_57/MatMulMatMuldense_56/Relu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_57/MatMul?
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_57/BiasAdd/ReadVariableOp?
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_57/BiasAdds
dense_57/ReluReludense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_57/Reluy
dropout_31/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_31/dropout/Const?
dropout_31/dropout/MulMuldense_57/Relu:activations:0!dropout_31/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout_31/dropout/Mul
dropout_31/dropout/ShapeShapedense_57/Relu:activations:0*
T0*
_output_shapes
:2
dropout_31/dropout/Shape?
/dropout_31/dropout/random_uniform/RandomUniformRandomUniform!dropout_31/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype021
/dropout_31/dropout/random_uniform/RandomUniform?
!dropout_31/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_31/dropout/GreaterEqual/y?
dropout_31/dropout/GreaterEqualGreaterEqual8dropout_31/dropout/random_uniform/RandomUniform:output:0*dropout_31/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2!
dropout_31/dropout/GreaterEqual?
dropout_31/dropout/CastCast#dropout_31/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_31/dropout/Cast?
dropout_31/dropout/Mul_1Muldropout_31/dropout/Mul:z:0dropout_31/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout_31/dropout/Mul_1?
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_58/MatMul/ReadVariableOp?
dense_58/MatMulMatMuldropout_31/dropout/Mul_1:z:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_58/MatMul?
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_58/BiasAdd/ReadVariableOp?
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_58/BiasAdd|
dense_58/SigmoidSigmoiddense_58/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_58/Sigmoid?
2conv2d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_37/kernel/Regularizer/SquareSquare:conv2d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_37/kernel/Regularizer/Square?
"conv2d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_37/kernel/Regularizer/Const?
 conv2d_37/kernel/Regularizer/SumSum'conv2d_37/kernel/Regularizer/Square:y:0+conv2d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/Sum?
"conv2d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_37/kernel/Regularizer/mul/x?
 conv2d_37/kernel/Regularizer/mulMul+conv2d_37/kernel/Regularizer/mul/x:output:0)conv2d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/mul?
2conv2d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_38/kernel/Regularizer/SquareSquare:conv2d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_38/kernel/Regularizer/Square?
"conv2d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_38/kernel/Regularizer/Const?
 conv2d_38/kernel/Regularizer/SumSum'conv2d_38/kernel/Regularizer/Square:y:0+conv2d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/Sum?
"conv2d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_38/kernel/Regularizer/mul/x?
 conv2d_38/kernel/Regularizer/mulMul+conv2d_38/kernel/Regularizer/mul/x:output:0)conv2d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/mul?
1dense_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype023
1dense_55/kernel/Regularizer/Square/ReadVariableOp?
"dense_55/kernel/Regularizer/SquareSquare9dense_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??@2$
"dense_55/kernel/Regularizer/Square?
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_55/kernel/Regularizer/Const?
dense_55/kernel/Regularizer/SumSum&dense_55/kernel/Regularizer/Square:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/Sum?
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!dense_55/kernel/Regularizer/mul/x?
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/mul?
IdentityIdentitydense_58/Sigmoid:y:0!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp3^conv2d_37/kernel/Regularizer/Square/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp3^conv2d_38/kernel/Regularizer/Square/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp2^dense_55/kernel/Regularizer/Square/ReadVariableOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2h
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2h
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2f
1dense_55/kernel/Regularizer/Square/ReadVariableOp1dense_55/kernel/Regularizer/Square/ReadVariableOp2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_30_layer_call_fn_83107

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????UU * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_823252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????UU 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????UU 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????UU 
 
_user_specified_nameinputs
?
c
E__inference_dropout_30_layer_call_and_return_conditional_losses_82330

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????UU 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????UU 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????UU :W S
/
_output_shapes
:?????????UU 
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_82269

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_33_layer_call_fn_82275

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_822692
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_83634
file_prefix%
!assignvariableop_conv2d_37_kernel%
!assignvariableop_1_conv2d_37_bias'
#assignvariableop_2_conv2d_38_kernel%
!assignvariableop_3_conv2d_38_bias&
"assignvariableop_4_dense_55_kernel$
 assignvariableop_5_dense_55_bias&
"assignvariableop_6_dense_56_kernel$
 assignvariableop_7_dense_56_bias&
"assignvariableop_8_dense_57_kernel$
 assignvariableop_9_dense_57_bias'
#assignvariableop_10_dense_58_kernel%
!assignvariableop_11_dense_58_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1&
"assignvariableop_21_true_positives'
#assignvariableop_22_false_positives(
$assignvariableop_23_true_positives_1'
#assignvariableop_24_false_negatives/
+assignvariableop_25_adam_conv2d_37_kernel_m-
)assignvariableop_26_adam_conv2d_37_bias_m/
+assignvariableop_27_adam_conv2d_38_kernel_m-
)assignvariableop_28_adam_conv2d_38_bias_m.
*assignvariableop_29_adam_dense_55_kernel_m,
(assignvariableop_30_adam_dense_55_bias_m.
*assignvariableop_31_adam_dense_56_kernel_m,
(assignvariableop_32_adam_dense_56_bias_m.
*assignvariableop_33_adam_dense_57_kernel_m,
(assignvariableop_34_adam_dense_57_bias_m.
*assignvariableop_35_adam_dense_58_kernel_m,
(assignvariableop_36_adam_dense_58_bias_m/
+assignvariableop_37_adam_conv2d_37_kernel_v-
)assignvariableop_38_adam_conv2d_37_bias_v/
+assignvariableop_39_adam_conv2d_38_kernel_v-
)assignvariableop_40_adam_conv2d_38_bias_v.
*assignvariableop_41_adam_dense_55_kernel_v,
(assignvariableop_42_adam_dense_55_bias_v.
*assignvariableop_43_adam_dense_56_kernel_v,
(assignvariableop_44_adam_dense_56_bias_v.
*assignvariableop_45_adam_dense_57_kernel_v,
(assignvariableop_46_adam_dense_57_bias_v.
*assignvariableop_47_adam_dense_58_kernel_v,
(assignvariableop_48_adam_dense_58_bias_v
identity_50??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_37_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_37_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_38_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_38_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_55_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_55_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_56_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_56_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_57_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_57_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_58_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_58_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_true_positivesIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_false_positivesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp$assignvariableop_23_true_positives_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_negativesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_37_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_37_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_38_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_38_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_55_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_55_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_56_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_56_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_57_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_57_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_58_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_58_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_37_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_37_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_38_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_38_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_55_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_55_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_56_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_56_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_57_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_57_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_58_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_58_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49?	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
}
(__inference_dense_58_layer_call_fn_83274

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_58_layer_call_and_return_conditional_losses_825192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
}
(__inference_dense_57_layer_call_fn_83227

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_57_layer_call_and_return_conditional_losses_824622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?Q
?
H__inference_sequential_20_layer_call_and_return_conditional_losses_82671

inputs
conv2d_37_82617
conv2d_37_82619
conv2d_38_82624
conv2d_38_82626
dense_55_82631
dense_55_82633
dense_56_82636
dense_56_82638
dense_57_82641
dense_57_82643
dense_58_82647
dense_58_82649
identity??!conv2d_37/StatefulPartitionedCall?2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?!conv2d_38/StatefulPartitionedCall?2conv2d_38/kernel/Regularizer/Square/ReadVariableOp? dense_55/StatefulPartitionedCall?1dense_55/kernel/Regularizer/Square/ReadVariableOp? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall?"dropout_30/StatefulPartitionedCall?"dropout_31/StatefulPartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_37_82617conv2d_37_82619*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_822962#
!conv2d_37/StatefulPartitionedCall?
 max_pooling2d_32/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????UU * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_822572"
 max_pooling2d_32/PartitionedCall?
"dropout_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????UU * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_823252$
"dropout_30/StatefulPartitionedCall?
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall+dropout_30/StatefulPartitionedCall:output:0conv2d_38_82624conv2d_38_82626*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????SS0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_823602#
!conv2d_38/StatefulPartitionedCall?
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_822692"
 max_pooling2d_33/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_18_layer_call_and_return_conditional_losses_823832
flatten_18/PartitionedCall?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_55_82631dense_55_82633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_55_layer_call_and_return_conditional_losses_824082"
 dense_55/StatefulPartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0dense_56_82636dense_56_82638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_56_layer_call_and_return_conditional_losses_824352"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_82641dense_57_82643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_57_layer_call_and_return_conditional_losses_824622"
 dense_57/StatefulPartitionedCall?
"dropout_31/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0#^dropout_30/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_824902$
"dropout_31/StatefulPartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall+dropout_31/StatefulPartitionedCall:output:0dense_58_82647dense_58_82649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_58_layer_call_and_return_conditional_losses_825192"
 dense_58/StatefulPartitionedCall?
2conv2d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_37_82617*&
_output_shapes
: *
dtype024
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_37/kernel/Regularizer/SquareSquare:conv2d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_37/kernel/Regularizer/Square?
"conv2d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_37/kernel/Regularizer/Const?
 conv2d_37/kernel/Regularizer/SumSum'conv2d_37/kernel/Regularizer/Square:y:0+conv2d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/Sum?
"conv2d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_37/kernel/Regularizer/mul/x?
 conv2d_37/kernel/Regularizer/mulMul+conv2d_37/kernel/Regularizer/mul/x:output:0)conv2d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/mul?
2conv2d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_38_82624*&
_output_shapes
: 0*
dtype024
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_38/kernel/Regularizer/SquareSquare:conv2d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_38/kernel/Regularizer/Square?
"conv2d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_38/kernel/Regularizer/Const?
 conv2d_38/kernel/Regularizer/SumSum'conv2d_38/kernel/Regularizer/Square:y:0+conv2d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/Sum?
"conv2d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_38/kernel/Regularizer/mul/x?
 conv2d_38/kernel/Regularizer/mulMul+conv2d_38/kernel/Regularizer/mul/x:output:0)conv2d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/mul?
1dense_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_55_82631* 
_output_shapes
:
??@*
dtype023
1dense_55/kernel/Regularizer/Square/ReadVariableOp?
"dense_55/kernel/Regularizer/SquareSquare9dense_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??@2$
"dense_55/kernel/Regularizer/Square?
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_55/kernel/Regularizer/Const?
dense_55/kernel/Regularizer/SumSum&dense_55/kernel/Regularizer/Square:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/Sum?
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!dense_55/kernel/Regularizer/mul/x?
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/mul?
IdentityIdentity)dense_58/StatefulPartitionedCall:output:0"^conv2d_37/StatefulPartitionedCall3^conv2d_37/kernel/Regularizer/Square/ReadVariableOp"^conv2d_38/StatefulPartitionedCall3^conv2d_38/kernel/Regularizer/Square/ReadVariableOp!^dense_55/StatefulPartitionedCall2^dense_55/kernel/Regularizer/Square/ReadVariableOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall#^dropout_30/StatefulPartitionedCall#^dropout_31/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2h
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2h
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2f
1dense_55/kernel/Regularizer/Square/ReadVariableOp1dense_55/kernel/Regularizer/Square/ReadVariableOp2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2H
"dropout_30/StatefulPartitionedCall"dropout_30/StatefulPartitionedCall2H
"dropout_31/StatefulPartitionedCall"dropout_31/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
}
(__inference_dense_55_layer_call_fn_83187

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_55_layer_call_and_return_conditional_losses_824082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_31_layer_call_fn_83254

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_824952
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
-__inference_sequential_20_layer_call_fn_82784
conv2d_37_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_37_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_20_layer_call_and_return_conditional_losses_827572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_37_input
?
?
C__inference_dense_55_layer_call_and_return_conditional_losses_82408

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_55/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
1dense_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype023
1dense_55/kernel/Regularizer/Square/ReadVariableOp?
"dense_55/kernel/Regularizer/SquareSquare9dense_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??@2$
"dense_55/kernel/Regularizer/Square?
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_55/kernel/Regularizer/Const?
dense_55/kernel/Regularizer/SumSum&dense_55/kernel/Regularizer/Square:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/Sum?
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!dense_55/kernel/Regularizer/mul/x?
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_55/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_55/kernel/Regularizer/Square/ReadVariableOp1dense_55/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_82257

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_31_layer_call_fn_83249

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_824902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
C__inference_dense_57_layer_call_and_return_conditional_losses_83218

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
E__inference_dropout_31_layer_call_and_return_conditional_losses_82490

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
C__inference_dense_57_layer_call_and_return_conditional_losses_82462

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?Q
?
H__inference_sequential_20_layer_call_and_return_conditional_losses_82554
conv2d_37_input
conv2d_37_82307
conv2d_37_82309
conv2d_38_82371
conv2d_38_82373
dense_55_82419
dense_55_82421
dense_56_82446
dense_56_82448
dense_57_82473
dense_57_82475
dense_58_82530
dense_58_82532
identity??!conv2d_37/StatefulPartitionedCall?2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?!conv2d_38/StatefulPartitionedCall?2conv2d_38/kernel/Regularizer/Square/ReadVariableOp? dense_55/StatefulPartitionedCall?1dense_55/kernel/Regularizer/Square/ReadVariableOp? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall?"dropout_30/StatefulPartitionedCall?"dropout_31/StatefulPartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCallconv2d_37_inputconv2d_37_82307conv2d_37_82309*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_822962#
!conv2d_37/StatefulPartitionedCall?
 max_pooling2d_32/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????UU * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_822572"
 max_pooling2d_32/PartitionedCall?
"dropout_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????UU * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_823252$
"dropout_30/StatefulPartitionedCall?
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall+dropout_30/StatefulPartitionedCall:output:0conv2d_38_82371conv2d_38_82373*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????SS0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_823602#
!conv2d_38/StatefulPartitionedCall?
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_822692"
 max_pooling2d_33/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_18_layer_call_and_return_conditional_losses_823832
flatten_18/PartitionedCall?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_55_82419dense_55_82421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_55_layer_call_and_return_conditional_losses_824082"
 dense_55/StatefulPartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0dense_56_82446dense_56_82448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_56_layer_call_and_return_conditional_losses_824352"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_82473dense_57_82475*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_57_layer_call_and_return_conditional_losses_824622"
 dense_57/StatefulPartitionedCall?
"dropout_31/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0#^dropout_30/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_824902$
"dropout_31/StatefulPartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall+dropout_31/StatefulPartitionedCall:output:0dense_58_82530dense_58_82532*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_58_layer_call_and_return_conditional_losses_825192"
 dense_58/StatefulPartitionedCall?
2conv2d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_37_82307*&
_output_shapes
: *
dtype024
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_37/kernel/Regularizer/SquareSquare:conv2d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_37/kernel/Regularizer/Square?
"conv2d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_37/kernel/Regularizer/Const?
 conv2d_37/kernel/Regularizer/SumSum'conv2d_37/kernel/Regularizer/Square:y:0+conv2d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/Sum?
"conv2d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_37/kernel/Regularizer/mul/x?
 conv2d_37/kernel/Regularizer/mulMul+conv2d_37/kernel/Regularizer/mul/x:output:0)conv2d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/mul?
2conv2d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_38_82371*&
_output_shapes
: 0*
dtype024
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_38/kernel/Regularizer/SquareSquare:conv2d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_38/kernel/Regularizer/Square?
"conv2d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_38/kernel/Regularizer/Const?
 conv2d_38/kernel/Regularizer/SumSum'conv2d_38/kernel/Regularizer/Square:y:0+conv2d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/Sum?
"conv2d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_38/kernel/Regularizer/mul/x?
 conv2d_38/kernel/Regularizer/mulMul+conv2d_38/kernel/Regularizer/mul/x:output:0)conv2d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/mul?
1dense_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_55_82419* 
_output_shapes
:
??@*
dtype023
1dense_55/kernel/Regularizer/Square/ReadVariableOp?
"dense_55/kernel/Regularizer/SquareSquare9dense_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??@2$
"dense_55/kernel/Regularizer/Square?
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_55/kernel/Regularizer/Const?
dense_55/kernel/Regularizer/SumSum&dense_55/kernel/Regularizer/Square:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/Sum?
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!dense_55/kernel/Regularizer/mul/x?
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/mul?
IdentityIdentity)dense_58/StatefulPartitionedCall:output:0"^conv2d_37/StatefulPartitionedCall3^conv2d_37/kernel/Regularizer/Square/ReadVariableOp"^conv2d_38/StatefulPartitionedCall3^conv2d_38/kernel/Regularizer/Square/ReadVariableOp!^dense_55/StatefulPartitionedCall2^dense_55/kernel/Regularizer/Square/ReadVariableOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall#^dropout_30/StatefulPartitionedCall#^dropout_31/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2h
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2h
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2f
1dense_55/kernel/Regularizer/Square/ReadVariableOp1dense_55/kernel/Regularizer/Square/ReadVariableOp2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2H
"dropout_30/StatefulPartitionedCall"dropout_30/StatefulPartitionedCall2H
"dropout_31/StatefulPartitionedCall"dropout_31/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_37_input
?
d
E__inference_dropout_30_layer_call_and_return_conditional_losses_82325

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????UU 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????UU *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????UU 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????UU 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????UU 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????UU 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????UU :W S
/
_output_shapes
:?????????UU 
 
_user_specified_nameinputs
?N
?
H__inference_sequential_20_layer_call_and_return_conditional_losses_82611
conv2d_37_input
conv2d_37_82557
conv2d_37_82559
conv2d_38_82564
conv2d_38_82566
dense_55_82571
dense_55_82573
dense_56_82576
dense_56_82578
dense_57_82581
dense_57_82583
dense_58_82587
dense_58_82589
identity??!conv2d_37/StatefulPartitionedCall?2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?!conv2d_38/StatefulPartitionedCall?2conv2d_38/kernel/Regularizer/Square/ReadVariableOp? dense_55/StatefulPartitionedCall?1dense_55/kernel/Regularizer/Square/ReadVariableOp? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCallconv2d_37_inputconv2d_37_82557conv2d_37_82559*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_822962#
!conv2d_37/StatefulPartitionedCall?
 max_pooling2d_32/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????UU * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_822572"
 max_pooling2d_32/PartitionedCall?
dropout_30/PartitionedCallPartitionedCall)max_pooling2d_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????UU * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_823302
dropout_30/PartitionedCall?
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0conv2d_38_82564conv2d_38_82566*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????SS0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_823602#
!conv2d_38/StatefulPartitionedCall?
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_822692"
 max_pooling2d_33/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_18_layer_call_and_return_conditional_losses_823832
flatten_18/PartitionedCall?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_55_82571dense_55_82573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_55_layer_call_and_return_conditional_losses_824082"
 dense_55/StatefulPartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0dense_56_82576dense_56_82578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_56_layer_call_and_return_conditional_losses_824352"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_82581dense_57_82583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_57_layer_call_and_return_conditional_losses_824622"
 dense_57/StatefulPartitionedCall?
dropout_31/PartitionedCallPartitionedCall)dense_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_824952
dropout_31/PartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall#dropout_31/PartitionedCall:output:0dense_58_82587dense_58_82589*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_58_layer_call_and_return_conditional_losses_825192"
 dense_58/StatefulPartitionedCall?
2conv2d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_37_82557*&
_output_shapes
: *
dtype024
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_37/kernel/Regularizer/SquareSquare:conv2d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_37/kernel/Regularizer/Square?
"conv2d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_37/kernel/Regularizer/Const?
 conv2d_37/kernel/Regularizer/SumSum'conv2d_37/kernel/Regularizer/Square:y:0+conv2d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/Sum?
"conv2d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_37/kernel/Regularizer/mul/x?
 conv2d_37/kernel/Regularizer/mulMul+conv2d_37/kernel/Regularizer/mul/x:output:0)conv2d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/mul?
2conv2d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_38_82564*&
_output_shapes
: 0*
dtype024
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_38/kernel/Regularizer/SquareSquare:conv2d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_38/kernel/Regularizer/Square?
"conv2d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_38/kernel/Regularizer/Const?
 conv2d_38/kernel/Regularizer/SumSum'conv2d_38/kernel/Regularizer/Square:y:0+conv2d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/Sum?
"conv2d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_38/kernel/Regularizer/mul/x?
 conv2d_38/kernel/Regularizer/mulMul+conv2d_38/kernel/Regularizer/mul/x:output:0)conv2d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/mul?
1dense_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_55_82571* 
_output_shapes
:
??@*
dtype023
1dense_55/kernel/Regularizer/Square/ReadVariableOp?
"dense_55/kernel/Regularizer/SquareSquare9dense_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??@2$
"dense_55/kernel/Regularizer/Square?
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_55/kernel/Regularizer/Const?
dense_55/kernel/Regularizer/SumSum&dense_55/kernel/Regularizer/Square:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/Sum?
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!dense_55/kernel/Regularizer/mul/x?
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/mul?
IdentityIdentity)dense_58/StatefulPartitionedCall:output:0"^conv2d_37/StatefulPartitionedCall3^conv2d_37/kernel/Regularizer/Square/ReadVariableOp"^conv2d_38/StatefulPartitionedCall3^conv2d_38/kernel/Regularizer/Square/ReadVariableOp!^dense_55/StatefulPartitionedCall2^dense_55/kernel/Regularizer/Square/ReadVariableOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2h
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2h
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2f
1dense_55/kernel/Regularizer/Square/ReadVariableOp1dense_55/kernel/Regularizer/Square/ReadVariableOp2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_37_input
?a
?	
H__inference_sequential_20_layer_call_and_return_conditional_losses_82995

inputs,
(conv2d_37_conv2d_readvariableop_resource-
)conv2d_37_biasadd_readvariableop_resource,
(conv2d_38_conv2d_readvariableop_resource-
)conv2d_38_biasadd_readvariableop_resource+
'dense_55_matmul_readvariableop_resource,
(dense_55_biasadd_readvariableop_resource+
'dense_56_matmul_readvariableop_resource,
(dense_56_biasadd_readvariableop_resource+
'dense_57_matmul_readvariableop_resource,
(dense_57_biasadd_readvariableop_resource+
'dense_58_matmul_readvariableop_resource,
(dense_58_biasadd_readvariableop_resource
identity?? conv2d_37/BiasAdd/ReadVariableOp?conv2d_37/Conv2D/ReadVariableOp?2conv2d_37/kernel/Regularizer/Square/ReadVariableOp? conv2d_38/BiasAdd/ReadVariableOp?conv2d_38/Conv2D/ReadVariableOp?2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?dense_55/BiasAdd/ReadVariableOp?dense_55/MatMul/ReadVariableOp?1dense_55/kernel/Regularizer/Square/ReadVariableOp?dense_56/BiasAdd/ReadVariableOp?dense_56/MatMul/ReadVariableOp?dense_57/BiasAdd/ReadVariableOp?dense_57/MatMul/ReadVariableOp?dense_58/BiasAdd/ReadVariableOp?dense_58/MatMul/ReadVariableOp?
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_37/Conv2D/ReadVariableOp?
conv2d_37/Conv2DConv2Dinputs'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
conv2d_37/Conv2D?
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp?
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_37/BiasAdd?
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_37/Relu?
max_pooling2d_32/MaxPoolMaxPoolconv2d_37/Relu:activations:0*/
_output_shapes
:?????????UU *
ksize
*
paddingVALID*
strides
2
max_pooling2d_32/MaxPool?
dropout_30/IdentityIdentity!max_pooling2d_32/MaxPool:output:0*
T0*/
_output_shapes
:?????????UU 2
dropout_30/Identity?
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02!
conv2d_38/Conv2D/ReadVariableOp?
conv2d_38/Conv2DConv2Ddropout_30/Identity:output:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????SS0*
paddingVALID*
strides
2
conv2d_38/Conv2D?
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp?
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????SS02
conv2d_38/BiasAdd~
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*/
_output_shapes
:?????????SS02
conv2d_38/Relu?
max_pooling2d_33/MaxPoolMaxPoolconv2d_38/Relu:activations:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_33/MaxPoolu
flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  2
flatten_18/Const?
flatten_18/ReshapeReshape!max_pooling2d_33/MaxPool:output:0flatten_18/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_18/Reshape?
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02 
dense_55/MatMul/ReadVariableOp?
dense_55/MatMulMatMulflatten_18/Reshape:output:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_55/MatMul?
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_55/BiasAdd/ReadVariableOp?
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_55/BiasAdds
dense_55/ReluReludense_55/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_55/Relu?
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_56/MatMul/ReadVariableOp?
dense_56/MatMulMatMuldense_55/Relu:activations:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_56/MatMul?
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_56/BiasAdd/ReadVariableOp?
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_56/BiasAdds
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_56/Relu?
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_57/MatMul/ReadVariableOp?
dense_57/MatMulMatMuldense_56/Relu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_57/MatMul?
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_57/BiasAdd/ReadVariableOp?
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_57/BiasAdds
dense_57/ReluReludense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_57/Relu?
dropout_31/IdentityIdentitydense_57/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout_31/Identity?
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_58/MatMul/ReadVariableOp?
dense_58/MatMulMatMuldropout_31/Identity:output:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_58/MatMul?
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_58/BiasAdd/ReadVariableOp?
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_58/BiasAdd|
dense_58/SigmoidSigmoiddense_58/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_58/Sigmoid?
2conv2d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_37/kernel/Regularizer/SquareSquare:conv2d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_37/kernel/Regularizer/Square?
"conv2d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_37/kernel/Regularizer/Const?
 conv2d_37/kernel/Regularizer/SumSum'conv2d_37/kernel/Regularizer/Square:y:0+conv2d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/Sum?
"conv2d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_37/kernel/Regularizer/mul/x?
 conv2d_37/kernel/Regularizer/mulMul+conv2d_37/kernel/Regularizer/mul/x:output:0)conv2d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/mul?
2conv2d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_38/kernel/Regularizer/SquareSquare:conv2d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_38/kernel/Regularizer/Square?
"conv2d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_38/kernel/Regularizer/Const?
 conv2d_38/kernel/Regularizer/SumSum'conv2d_38/kernel/Regularizer/Square:y:0+conv2d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/Sum?
"conv2d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_38/kernel/Regularizer/mul/x?
 conv2d_38/kernel/Regularizer/mulMul+conv2d_38/kernel/Regularizer/mul/x:output:0)conv2d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/mul?
1dense_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype023
1dense_55/kernel/Regularizer/Square/ReadVariableOp?
"dense_55/kernel/Regularizer/SquareSquare9dense_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??@2$
"dense_55/kernel/Regularizer/Square?
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_55/kernel/Regularizer/Const?
dense_55/kernel/Regularizer/SumSum&dense_55/kernel/Regularizer/Square:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/Sum?
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!dense_55/kernel/Regularizer/mul/x?
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/mul?
IdentityIdentitydense_58/Sigmoid:y:0!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp3^conv2d_37/kernel/Regularizer/Square/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp3^conv2d_38/kernel/Regularizer/Square/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp2^dense_55/kernel/Regularizer/Square/ReadVariableOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2h
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2h
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2f
1dense_55/kernel/Regularizer/Square/ReadVariableOp1dense_55/kernel/Regularizer/Square/ReadVariableOp2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?c
?
__inference__traced_save_83477
file_prefix/
+savev2_conv2d_37_kernel_read_readvariableop-
)savev2_conv2d_37_bias_read_readvariableop/
+savev2_conv2d_38_kernel_read_readvariableop-
)savev2_conv2d_38_bias_read_readvariableop.
*savev2_dense_55_kernel_read_readvariableop,
(savev2_dense_55_bias_read_readvariableop.
*savev2_dense_56_kernel_read_readvariableop,
(savev2_dense_56_bias_read_readvariableop.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableop.
*savev2_dense_58_kernel_read_readvariableop,
(savev2_dense_58_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop6
2savev2_adam_conv2d_37_kernel_m_read_readvariableop4
0savev2_adam_conv2d_37_bias_m_read_readvariableop6
2savev2_adam_conv2d_38_kernel_m_read_readvariableop4
0savev2_adam_conv2d_38_bias_m_read_readvariableop5
1savev2_adam_dense_55_kernel_m_read_readvariableop3
/savev2_adam_dense_55_bias_m_read_readvariableop5
1savev2_adam_dense_56_kernel_m_read_readvariableop3
/savev2_adam_dense_56_bias_m_read_readvariableop5
1savev2_adam_dense_57_kernel_m_read_readvariableop3
/savev2_adam_dense_57_bias_m_read_readvariableop5
1savev2_adam_dense_58_kernel_m_read_readvariableop3
/savev2_adam_dense_58_bias_m_read_readvariableop6
2savev2_adam_conv2d_37_kernel_v_read_readvariableop4
0savev2_adam_conv2d_37_bias_v_read_readvariableop6
2savev2_adam_conv2d_38_kernel_v_read_readvariableop4
0savev2_adam_conv2d_38_bias_v_read_readvariableop5
1savev2_adam_dense_55_kernel_v_read_readvariableop3
/savev2_adam_dense_55_bias_v_read_readvariableop5
1savev2_adam_dense_56_kernel_v_read_readvariableop3
/savev2_adam_dense_56_bias_v_read_readvariableop5
1savev2_adam_dense_57_kernel_v_read_readvariableop3
/savev2_adam_dense_57_bias_v_read_readvariableop5
1savev2_adam_dense_58_kernel_v_read_readvariableop3
/savev2_adam_dense_58_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableop+savev2_conv2d_38_kernel_read_readvariableop)savev2_conv2d_38_bias_read_readvariableop*savev2_dense_55_kernel_read_readvariableop(savev2_dense_55_bias_read_readvariableop*savev2_dense_56_kernel_read_readvariableop(savev2_dense_56_bias_read_readvariableop*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableop*savev2_dense_58_kernel_read_readvariableop(savev2_dense_58_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop2savev2_adam_conv2d_37_kernel_m_read_readvariableop0savev2_adam_conv2d_37_bias_m_read_readvariableop2savev2_adam_conv2d_38_kernel_m_read_readvariableop0savev2_adam_conv2d_38_bias_m_read_readvariableop1savev2_adam_dense_55_kernel_m_read_readvariableop/savev2_adam_dense_55_bias_m_read_readvariableop1savev2_adam_dense_56_kernel_m_read_readvariableop/savev2_adam_dense_56_bias_m_read_readvariableop1savev2_adam_dense_57_kernel_m_read_readvariableop/savev2_adam_dense_57_bias_m_read_readvariableop1savev2_adam_dense_58_kernel_m_read_readvariableop/savev2_adam_dense_58_bias_m_read_readvariableop2savev2_adam_conv2d_37_kernel_v_read_readvariableop0savev2_adam_conv2d_37_bias_v_read_readvariableop2savev2_adam_conv2d_38_kernel_v_read_readvariableop0savev2_adam_conv2d_38_bias_v_read_readvariableop1savev2_adam_dense_55_kernel_v_read_readvariableop/savev2_adam_dense_55_bias_v_read_readvariableop1savev2_adam_dense_56_kernel_v_read_readvariableop/savev2_adam_dense_56_bias_v_read_readvariableop1savev2_adam_dense_57_kernel_v_read_readvariableop/savev2_adam_dense_57_bias_v_read_readvariableop1savev2_adam_dense_58_kernel_v_read_readvariableop/savev2_adam_dense_58_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : 0:0:
??@:@:@@:@:@@:@:@:: : : : : : : : : ::::: : : 0:0:
??@:@:@@:@:@@:@:@:: : : 0:0:
??@:@:@@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:&"
 
_output_shapes
:
??@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$	 

_output_shapes

:@@: 


_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:&"
 
_output_shapes
:
??@: 

_output_shapes
:@:$  

_output_shapes

:@@: !

_output_shapes
:@:$" 

_output_shapes

:@@: #

_output_shapes
:@:$$ 

_output_shapes

:@: %

_output_shapes
::,&(
&
_output_shapes
: : '

_output_shapes
: :,((
&
_output_shapes
: 0: )

_output_shapes
:0:&*"
 
_output_shapes
:
??@: +

_output_shapes
:@:$, 

_output_shapes

:@@: -

_output_shapes
:@:$. 

_output_shapes

:@@: /

_output_shapes
:@:$0 

_output_shapes

:@: 1

_output_shapes
::2

_output_shapes
: 
?
?
__inference_loss_fn_0_83285?
;conv2d_37_kernel_regularizer_square_readvariableop_resource
identity??2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
2conv2d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_37_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_37/kernel/Regularizer/SquareSquare:conv2d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_37/kernel/Regularizer/Square?
"conv2d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_37/kernel/Regularizer/Const?
 conv2d_37/kernel/Regularizer/SumSum'conv2d_37/kernel/Regularizer/Square:y:0+conv2d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/Sum?
"conv2d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_37/kernel/Regularizer/mul/x?
 conv2d_37/kernel/Regularizer/mulMul+conv2d_37/kernel/Regularizer/mul/x:output:0)conv2d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/mul?
IdentityIdentity$conv2d_37/kernel/Regularizer/mul:z:03^conv2d_37/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2conv2d_37/kernel/Regularizer/Square/ReadVariableOp
?
?
D__inference_conv2d_38_layer_call_and_return_conditional_losses_83135

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????SS0*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????SS02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????SS02
Relu?
2conv2d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_38/kernel/Regularizer/SquareSquare:conv2d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_38/kernel/Regularizer/Square?
"conv2d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_38/kernel/Regularizer/Const?
 conv2d_38/kernel/Regularizer/SumSum'conv2d_38/kernel/Regularizer/Square:y:0+conv2d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/Sum?
"conv2d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_38/kernel/Regularizer/mul/x?
 conv2d_38/kernel/Regularizer/mulMul+conv2d_38/kernel/Regularizer/mul/x:output:0)conv2d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_38/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????SS02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????UU ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2conv2d_38/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????UU 
 
_user_specified_nameinputs
?	
?
C__inference_dense_58_layer_call_and_return_conditional_losses_83265

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_82841
conv2d_37_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_37_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_822512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_37_input
?
?
D__inference_conv2d_37_layer_call_and_return_conditional_losses_82296

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
2conv2d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_37/kernel/Regularizer/SquareSquare:conv2d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_37/kernel/Regularizer/Square?
"conv2d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_37/kernel/Regularizer/Const?
 conv2d_37/kernel/Regularizer/SumSum'conv2d_37/kernel/Regularizer/Square:y:0+conv2d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/Sum?
"conv2d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_37/kernel/Regularizer/mul/x?
 conv2d_37/kernel/Regularizer/mulMul+conv2d_37/kernel/Regularizer/mul/x:output:0)conv2d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_37/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2conv2d_37/kernel/Regularizer/Square/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_83296?
;conv2d_38_kernel_regularizer_square_readvariableop_resource
identity??2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
2conv2d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_38_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_38/kernel/Regularizer/SquareSquare:conv2d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_38/kernel/Regularizer/Square?
"conv2d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_38/kernel/Regularizer/Const?
 conv2d_38/kernel/Regularizer/SumSum'conv2d_38/kernel/Regularizer/Square:y:0+conv2d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/Sum?
"conv2d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_38/kernel/Regularizer/mul/x?
 conv2d_38/kernel/Regularizer/mulMul+conv2d_38/kernel/Regularizer/mul/x:output:0)conv2d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/mul?
IdentityIdentity$conv2d_38/kernel/Regularizer/mul:z:03^conv2d_38/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2conv2d_38/kernel/Regularizer/Square/ReadVariableOp
?	
?
-__inference_sequential_20_layer_call_fn_83053

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_20_layer_call_and_return_conditional_losses_827572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_30_layer_call_fn_83112

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????UU * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_823302
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????UU 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????UU :W S
/
_output_shapes
:?????????UU 
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_32_layer_call_fn_82263

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_822572
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_31_layer_call_and_return_conditional_losses_83244

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
C__inference_dense_56_layer_call_and_return_conditional_losses_83198

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
E__inference_dropout_31_layer_call_and_return_conditional_losses_83239

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
-__inference_sequential_20_layer_call_fn_82698
conv2d_37_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_37_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_20_layer_call_and_return_conditional_losses_826712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_37_input
?
}
(__inference_dense_56_layer_call_fn_83207

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_56_layer_call_and_return_conditional_losses_824352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
c
E__inference_dropout_31_layer_call_and_return_conditional_losses_82495

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
-__inference_sequential_20_layer_call_fn_83024

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_20_layer_call_and_return_conditional_losses_826712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_56_layer_call_and_return_conditional_losses_82435

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_dense_55_layer_call_and_return_conditional_losses_83178

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_55/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
1dense_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype023
1dense_55/kernel/Regularizer/Square/ReadVariableOp?
"dense_55/kernel/Regularizer/SquareSquare9dense_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??@2$
"dense_55/kernel/Regularizer/Square?
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_55/kernel/Regularizer/Const?
dense_55/kernel/Regularizer/SumSum&dense_55/kernel/Regularizer/Square:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/Sum?
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!dense_55/kernel/Regularizer/mul/x?
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_55/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_55/kernel/Regularizer/Square/ReadVariableOp1dense_55/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_38_layer_call_and_return_conditional_losses_82360

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????SS0*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????SS02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????SS02
Relu?
2conv2d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_38/kernel/Regularizer/SquareSquare:conv2d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_38/kernel/Regularizer/Square?
"conv2d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_38/kernel/Regularizer/Const?
 conv2d_38/kernel/Regularizer/SumSum'conv2d_38/kernel/Regularizer/Square:y:0+conv2d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/Sum?
"conv2d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_38/kernel/Regularizer/mul/x?
 conv2d_38/kernel/Regularizer/mulMul+conv2d_38/kernel/Regularizer/mul/x:output:0)conv2d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_38/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????SS02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????UU ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2conv2d_38/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????UU 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_83307>
:dense_55_kernel_regularizer_square_readvariableop_resource
identity??1dense_55/kernel/Regularizer/Square/ReadVariableOp?
1dense_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_55_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??@*
dtype023
1dense_55/kernel/Regularizer/Square/ReadVariableOp?
"dense_55/kernel/Regularizer/SquareSquare9dense_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??@2$
"dense_55/kernel/Regularizer/Square?
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_55/kernel/Regularizer/Const?
dense_55/kernel/Regularizer/SumSum&dense_55/kernel/Regularizer/Square:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/Sum?
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!dense_55/kernel/Regularizer/mul/x?
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/mul?
IdentityIdentity#dense_55/kernel/Regularizer/mul:z:02^dense_55/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_55/kernel/Regularizer/Square/ReadVariableOp1dense_55/kernel/Regularizer/Square/ReadVariableOp
?
d
E__inference_dropout_30_layer_call_and_return_conditional_losses_83097

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????UU 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????UU *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????UU 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????UU 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????UU 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????UU 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????UU :W S
/
_output_shapes
:?????????UU 
 
_user_specified_nameinputs
?N
?
H__inference_sequential_20_layer_call_and_return_conditional_losses_82757

inputs
conv2d_37_82703
conv2d_37_82705
conv2d_38_82710
conv2d_38_82712
dense_55_82717
dense_55_82719
dense_56_82722
dense_56_82724
dense_57_82727
dense_57_82729
dense_58_82733
dense_58_82735
identity??!conv2d_37/StatefulPartitionedCall?2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?!conv2d_38/StatefulPartitionedCall?2conv2d_38/kernel/Regularizer/Square/ReadVariableOp? dense_55/StatefulPartitionedCall?1dense_55/kernel/Regularizer/Square/ReadVariableOp? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_37_82703conv2d_37_82705*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_822962#
!conv2d_37/StatefulPartitionedCall?
 max_pooling2d_32/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????UU * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_822572"
 max_pooling2d_32/PartitionedCall?
dropout_30/PartitionedCallPartitionedCall)max_pooling2d_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????UU * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_823302
dropout_30/PartitionedCall?
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0conv2d_38_82710conv2d_38_82712*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????SS0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_823602#
!conv2d_38/StatefulPartitionedCall?
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_822692"
 max_pooling2d_33/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)max_pooling2d_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_18_layer_call_and_return_conditional_losses_823832
flatten_18/PartitionedCall?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall#flatten_18/PartitionedCall:output:0dense_55_82717dense_55_82719*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_55_layer_call_and_return_conditional_losses_824082"
 dense_55/StatefulPartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0dense_56_82722dense_56_82724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_56_layer_call_and_return_conditional_losses_824352"
 dense_56/StatefulPartitionedCall?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_82727dense_57_82729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_57_layer_call_and_return_conditional_losses_824622"
 dense_57/StatefulPartitionedCall?
dropout_31/PartitionedCallPartitionedCall)dense_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_824952
dropout_31/PartitionedCall?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall#dropout_31/PartitionedCall:output:0dense_58_82733dense_58_82735*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_58_layer_call_and_return_conditional_losses_825192"
 dense_58/StatefulPartitionedCall?
2conv2d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_37_82703*&
_output_shapes
: *
dtype024
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_37/kernel/Regularizer/SquareSquare:conv2d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_37/kernel/Regularizer/Square?
"conv2d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_37/kernel/Regularizer/Const?
 conv2d_37/kernel/Regularizer/SumSum'conv2d_37/kernel/Regularizer/Square:y:0+conv2d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/Sum?
"conv2d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_37/kernel/Regularizer/mul/x?
 conv2d_37/kernel/Regularizer/mulMul+conv2d_37/kernel/Regularizer/mul/x:output:0)conv2d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_37/kernel/Regularizer/mul?
2conv2d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_38_82710*&
_output_shapes
: 0*
dtype024
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_38/kernel/Regularizer/SquareSquare:conv2d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_38/kernel/Regularizer/Square?
"conv2d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_38/kernel/Regularizer/Const?
 conv2d_38/kernel/Regularizer/SumSum'conv2d_38/kernel/Regularizer/Square:y:0+conv2d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/Sum?
"conv2d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_38/kernel/Regularizer/mul/x?
 conv2d_38/kernel/Regularizer/mulMul+conv2d_38/kernel/Regularizer/mul/x:output:0)conv2d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_38/kernel/Regularizer/mul?
1dense_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_55_82717* 
_output_shapes
:
??@*
dtype023
1dense_55/kernel/Regularizer/Square/ReadVariableOp?
"dense_55/kernel/Regularizer/SquareSquare9dense_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??@2$
"dense_55/kernel/Regularizer/Square?
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_55/kernel/Regularizer/Const?
dense_55/kernel/Regularizer/SumSum&dense_55/kernel/Regularizer/Square:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/Sum?
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2#
!dense_55/kernel/Regularizer/mul/x?
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_55/kernel/Regularizer/mul?
IdentityIdentity)dense_58/StatefulPartitionedCall:output:0"^conv2d_37/StatefulPartitionedCall3^conv2d_37/kernel/Regularizer/Square/ReadVariableOp"^conv2d_38/StatefulPartitionedCall3^conv2d_38/kernel/Regularizer/Square/ReadVariableOp!^dense_55/StatefulPartitionedCall2^dense_55/kernel/Regularizer/Square/ReadVariableOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2h
2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2conv2d_37/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2h
2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2conv2d_38/kernel/Regularizer/Square/ReadVariableOp2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2f
1dense_55/kernel/Regularizer/Square/ReadVariableOp1dense_55/kernel/Regularizer/Square/ReadVariableOp2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
conv2d_37_inputB
!serving_default_conv2d_37_input:0???????????<
dense_580
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?T
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?Q
_tf_keras_sequential?P{"class_name": "Sequential", "name": "sequential_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_20", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_37_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_32", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_30", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_31", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_20", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_37_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_32", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_30", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_31", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}, {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_37", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_32", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_30", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_30", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?


 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_38", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 85, 85, 32]}}
?
&	variables
'regularization_losses
(trainable_variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_55", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 37632}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 37632]}}
?

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_56", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_56", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_57", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_31", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_31", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_58", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_ratem?m? m?!m?.m?/m?4m?5m?:m?;m?Dm?Em?v?v? v?!v?.v?/v?4v?5v?:v?;v?Dv?Ev?"
	optimizer
v
0
1
 2
!3
.4
/5
46
57
:8
;9
D10
E11"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
v
0
1
 2
!3
.4
/5
46
57
:8
;9
D10
E11"
trackable_list_wrapper
?
Olayer_regularization_losses
	variables
regularization_losses
trainable_variables

Players
Qnon_trainable_variables
Rmetrics
Slayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_37/kernel
: 2conv2d_37/bias
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Tlayer_regularization_losses
	variables
regularization_losses
trainable_variables

Ulayers
Vnon_trainable_variables
Wmetrics
Xlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ylayer_regularization_losses
	variables
regularization_losses
trainable_variables

Zlayers
[non_trainable_variables
\metrics
]layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
^layer_regularization_losses
	variables
regularization_losses
trainable_variables

_layers
`non_trainable_variables
ametrics
blayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 02conv2d_38/kernel
:02conv2d_38/bias
.
 0
!1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
clayer_regularization_losses
"	variables
#regularization_losses
$trainable_variables

dlayers
enon_trainable_variables
fmetrics
glayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
hlayer_regularization_losses
&	variables
'regularization_losses
(trainable_variables

ilayers
jnon_trainable_variables
kmetrics
llayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mlayer_regularization_losses
*	variables
+regularization_losses
,trainable_variables

nlayers
onon_trainable_variables
pmetrics
qlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??@2dense_55/kernel
:@2dense_55/bias
.
.0
/1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
rlayer_regularization_losses
0	variables
1regularization_losses
2trainable_variables

slayers
tnon_trainable_variables
umetrics
vlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_56/kernel
:@2dense_56/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
wlayer_regularization_losses
6	variables
7regularization_losses
8trainable_variables

xlayers
ynon_trainable_variables
zmetrics
{layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_57/kernel
:@2dense_57/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
|layer_regularization_losses
<	variables
=regularization_losses
>trainable_variables

}layers
~non_trainable_variables
metrics
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
@	variables
Aregularization_losses
Btrainable_variables
?layers
?non_trainable_variables
?metrics
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_58/kernel
:2dense_58/bias
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
 ?layer_regularization_losses
F	variables
Gregularization_losses
Htrainable_variables
?layers
?non_trainable_variables
?metrics
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
?
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Adam/conv2d_37/kernel/m
!: 2Adam/conv2d_37/bias/m
/:- 02Adam/conv2d_38/kernel/m
!:02Adam/conv2d_38/bias/m
(:&
??@2Adam/dense_55/kernel/m
 :@2Adam/dense_55/bias/m
&:$@@2Adam/dense_56/kernel/m
 :@2Adam/dense_56/bias/m
&:$@@2Adam/dense_57/kernel/m
 :@2Adam/dense_57/bias/m
&:$@2Adam/dense_58/kernel/m
 :2Adam/dense_58/bias/m
/:- 2Adam/conv2d_37/kernel/v
!: 2Adam/conv2d_37/bias/v
/:- 02Adam/conv2d_38/kernel/v
!:02Adam/conv2d_38/bias/v
(:&
??@2Adam/dense_55/kernel/v
 :@2Adam/dense_55/bias/v
&:$@@2Adam/dense_56/kernel/v
 :@2Adam/dense_56/bias/v
&:$@@2Adam/dense_57/kernel/v
 :@2Adam/dense_57/bias/v
&:$@2Adam/dense_58/kernel/v
 :2Adam/dense_58/bias/v
?2?
 __inference__wrapped_model_82251?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0
conv2d_37_input???????????
?2?
H__inference_sequential_20_layer_call_and_return_conditional_losses_82554
H__inference_sequential_20_layer_call_and_return_conditional_losses_82925
H__inference_sequential_20_layer_call_and_return_conditional_losses_82995
H__inference_sequential_20_layer_call_and_return_conditional_losses_82611?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_20_layer_call_fn_82698
-__inference_sequential_20_layer_call_fn_83024
-__inference_sequential_20_layer_call_fn_83053
-__inference_sequential_20_layer_call_fn_82784?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_37_layer_call_and_return_conditional_losses_83076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_37_layer_call_fn_83085?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_82257?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_max_pooling2d_32_layer_call_fn_82263?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_dropout_30_layer_call_and_return_conditional_losses_83102
E__inference_dropout_30_layer_call_and_return_conditional_losses_83097?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_30_layer_call_fn_83107
*__inference_dropout_30_layer_call_fn_83112?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_38_layer_call_and_return_conditional_losses_83135?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_38_layer_call_fn_83144?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_82269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_max_pooling2d_33_layer_call_fn_82275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_flatten_18_layer_call_and_return_conditional_losses_83150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_18_layer_call_fn_83155?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_55_layer_call_and_return_conditional_losses_83178?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_55_layer_call_fn_83187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_56_layer_call_and_return_conditional_losses_83198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_56_layer_call_fn_83207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_57_layer_call_and_return_conditional_losses_83218?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_57_layer_call_fn_83227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dropout_31_layer_call_and_return_conditional_losses_83244
E__inference_dropout_31_layer_call_and_return_conditional_losses_83239?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_31_layer_call_fn_83254
*__inference_dropout_31_layer_call_fn_83249?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dense_58_layer_call_and_return_conditional_losses_83265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_58_layer_call_fn_83274?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_83285?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_83296?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_83307?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
#__inference_signature_wrapper_82841conv2d_37_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_82251? !./45:;DEB??
8?5
3?0
conv2d_37_input???????????
? "3?0
.
dense_58"?
dense_58??????????
D__inference_conv2d_37_layer_call_and_return_conditional_losses_83076p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
)__inference_conv2d_37_layer_call_fn_83085c9?6
/?,
*?'
inputs???????????
? ""???????????? ?
D__inference_conv2d_38_layer_call_and_return_conditional_losses_83135l !7?4
-?*
(?%
inputs?????????UU 
? "-?*
#? 
0?????????SS0
? ?
)__inference_conv2d_38_layer_call_fn_83144_ !7?4
-?*
(?%
inputs?????????UU 
? " ??????????SS0?
C__inference_dense_55_layer_call_and_return_conditional_losses_83178^./1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????@
? }
(__inference_dense_55_layer_call_fn_83187Q./1?.
'?$
"?
inputs???????????
? "??????????@?
C__inference_dense_56_layer_call_and_return_conditional_losses_83198\45/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? {
(__inference_dense_56_layer_call_fn_83207O45/?,
%?"
 ?
inputs?????????@
? "??????????@?
C__inference_dense_57_layer_call_and_return_conditional_losses_83218\:;/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? {
(__inference_dense_57_layer_call_fn_83227O:;/?,
%?"
 ?
inputs?????????@
? "??????????@?
C__inference_dense_58_layer_call_and_return_conditional_losses_83265\DE/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_dense_58_layer_call_fn_83274ODE/?,
%?"
 ?
inputs?????????@
? "???????????
E__inference_dropout_30_layer_call_and_return_conditional_losses_83097l;?8
1?.
(?%
inputs?????????UU 
p
? "-?*
#? 
0?????????UU 
? ?
E__inference_dropout_30_layer_call_and_return_conditional_losses_83102l;?8
1?.
(?%
inputs?????????UU 
p 
? "-?*
#? 
0?????????UU 
? ?
*__inference_dropout_30_layer_call_fn_83107_;?8
1?.
(?%
inputs?????????UU 
p
? " ??????????UU ?
*__inference_dropout_30_layer_call_fn_83112_;?8
1?.
(?%
inputs?????????UU 
p 
? " ??????????UU ?
E__inference_dropout_31_layer_call_and_return_conditional_losses_83239\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
E__inference_dropout_31_layer_call_and_return_conditional_losses_83244\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? }
*__inference_dropout_31_layer_call_fn_83249O3?0
)?&
 ?
inputs?????????@
p
? "??????????@}
*__inference_dropout_31_layer_call_fn_83254O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
E__inference_flatten_18_layer_call_and_return_conditional_losses_83150b7?4
-?*
(?%
inputs?????????0
? "'?$
?
0???????????
? ?
*__inference_flatten_18_layer_call_fn_83155U7?4
-?*
(?%
inputs?????????0
? "????????????:
__inference_loss_fn_0_83285?

? 
? "? :
__inference_loss_fn_1_83296 ?

? 
? "? :
__inference_loss_fn_2_83307.?

? 
? "? ?
K__inference_max_pooling2d_32_layer_call_and_return_conditional_losses_82257?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_32_layer_call_fn_82263?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_82269?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_33_layer_call_fn_82275?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_sequential_20_layer_call_and_return_conditional_losses_82554? !./45:;DEJ?G
@?=
3?0
conv2d_37_input???????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_20_layer_call_and_return_conditional_losses_82611? !./45:;DEJ?G
@?=
3?0
conv2d_37_input???????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_20_layer_call_and_return_conditional_losses_82925x !./45:;DEA?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_20_layer_call_and_return_conditional_losses_82995x !./45:;DEA?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
-__inference_sequential_20_layer_call_fn_82698t !./45:;DEJ?G
@?=
3?0
conv2d_37_input???????????
p

 
? "???????????
-__inference_sequential_20_layer_call_fn_82784t !./45:;DEJ?G
@?=
3?0
conv2d_37_input???????????
p 

 
? "???????????
-__inference_sequential_20_layer_call_fn_83024k !./45:;DEA?>
7?4
*?'
inputs???????????
p

 
? "???????????
-__inference_sequential_20_layer_call_fn_83053k !./45:;DEA?>
7?4
*?'
inputs???????????
p 

 
? "???????????
#__inference_signature_wrapper_82841? !./45:;DEU?R
? 
K?H
F
conv2d_37_input3?0
conv2d_37_input???????????"3?0
.
dense_58"?
dense_58?????????