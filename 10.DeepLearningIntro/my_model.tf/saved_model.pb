??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
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
executor_typestring ??
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
 ?"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??
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
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
?
!wide_and_deep_model/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!wide_and_deep_model/dense_21/bias
?
5wide_and_deep_model/dense_21/bias/Read/ReadVariableOpReadVariableOp!wide_and_deep_model/dense_21/bias*
_output_shapes
:*
dtype0
?
#wide_and_deep_model/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#wide_and_deep_model/dense_21/kernel
?
7wide_and_deep_model/dense_21/kernel/Read/ReadVariableOpReadVariableOp#wide_and_deep_model/dense_21/kernel*
_output_shapes

:*
dtype0
?
!wide_and_deep_model/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!wide_and_deep_model/dense_20/bias
?
5wide_and_deep_model/dense_20/bias/Read/ReadVariableOpReadVariableOp!wide_and_deep_model/dense_20/bias*
_output_shapes
:*
dtype0
?
#wide_and_deep_model/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*4
shared_name%#wide_and_deep_model/dense_20/kernel
?
7wide_and_deep_model/dense_20/kernel/Read/ReadVariableOpReadVariableOp#wide_and_deep_model/dense_20/kernel*
_output_shapes

:#*
dtype0
?
!wide_and_deep_model/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!wide_and_deep_model/dense_19/bias
?
5wide_and_deep_model/dense_19/bias/Read/ReadVariableOpReadVariableOp!wide_and_deep_model/dense_19/bias*
_output_shapes
:*
dtype0
?
#wide_and_deep_model/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#wide_and_deep_model/dense_19/kernel
?
7wide_and_deep_model/dense_19/kernel/Read/ReadVariableOpReadVariableOp#wide_and_deep_model/dense_19/kernel*
_output_shapes

:*
dtype0
?
!wide_and_deep_model/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!wide_and_deep_model/dense_18/bias
?
5wide_and_deep_model/dense_18/bias/Read/ReadVariableOpReadVariableOp!wide_and_deep_model/dense_18/bias*
_output_shapes
:*
dtype0
?
#wide_and_deep_model/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#wide_and_deep_model/dense_18/kernel
?
7wide_and_deep_model/dense_18/kernel/Read/ReadVariableOpReadVariableOp#wide_and_deep_model/dense_18/kernel*
_output_shapes

:*
dtype0

NoOpNoOp
?(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?(
value?(B?( B?(
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
hidden1
	hidden2

main_output

aux_output
	optimizer
loss

signatures*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

kernel
bias*
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

kernel
bias*
?
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

kernel
bias*
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

kernel
bias*
:
8iter
	9decay
:learning_rate
;momentum*
* 

<serving_default* 
c]
VARIABLE_VALUE#wide_and_deep_model/dense_18/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!wide_and_deep_model/dense_18/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#wide_and_deep_model/dense_19/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!wide_and_deep_model/dense_19/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#wide_and_deep_model/dense_20/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!wide_and_deep_model/dense_20/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#wide_and_deep_model/dense_21/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!wide_and_deep_model/dense_21/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*

=0
>1
?2*
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

Etrace_0* 

Ftrace_0* 

0
1*

0
1*
* 
?
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

Ltrace_0* 

Mtrace_0* 

0
1*

0
1*
* 
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

Strace_0* 

Ttrace_0* 

0
1*

0
1*
* 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
8
\	variables
]	keras_api
	^total
	_count*
8
`	variables
a	keras_api
	btotal
	ccount*
8
d	variables
e	keras_api
	ftotal
	gcount*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

^0
_1*

\	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

b0
c1*

`	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

f0
g1*

d	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2#wide_and_deep_model/dense_18/kernel!wide_and_deep_model/dense_18/bias#wide_and_deep_model/dense_19/kernel!wide_and_deep_model/dense_19/bias#wide_and_deep_model/dense_20/kernel!wide_and_deep_model/dense_20/bias#wide_and_deep_model/dense_21/kernel!wide_and_deep_model/dense_21/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_706509
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7wide_and_deep_model/dense_18/kernel/Read/ReadVariableOp5wide_and_deep_model/dense_18/bias/Read/ReadVariableOp7wide_and_deep_model/dense_19/kernel/Read/ReadVariableOp5wide_and_deep_model/dense_19/bias/Read/ReadVariableOp7wide_and_deep_model/dense_20/kernel/Read/ReadVariableOp5wide_and_deep_model/dense_20/bias/Read/ReadVariableOp7wide_and_deep_model/dense_21/kernel/Read/ReadVariableOp5wide_and_deep_model/dense_21/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_706724
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#wide_and_deep_model/dense_18/kernel!wide_and_deep_model/dense_18/bias#wide_and_deep_model/dense_19/kernel!wide_and_deep_model/dense_19/bias#wide_and_deep_model/dense_20/kernel!wide_and_deep_model/dense_20/bias#wide_and_deep_model/dense_21/kernel!wide_and_deep_model/dense_21/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotal_2count_2total_1count_1totalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_706788??
?'
?
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706567
inputs_0
inputs_19
'dense_18_matmul_readvariableop_resource:6
(dense_18_biasadd_readvariableop_resource:9
'dense_19_matmul_readvariableop_resource:6
(dense_19_biasadd_readvariableop_resource:9
'dense_20_matmul_readvariableop_resource:#6
(dense_20_biasadd_readvariableop_resource:9
'dense_21_matmul_readvariableop_resource:6
(dense_21_biasadd_readvariableop_resource:
identity

identity_1??dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:*
dtype0}
dense_18/MatMulMatMulinputs_1&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2inputs_0dense_19/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0?
dense_20/MatMulMatMulconcatenate/concat:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_21/MatMulMatMuldense_19/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_20/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????j

Identity_1Identitydense_21/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
)__inference_dense_19_layer_call_fn_706596

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_706324o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_dense_19_layer_call_and_return_conditional_losses_706324

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_20_layer_call_and_return_conditional_losses_706342

inputs0
matmul_readvariableop_resource:#-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:#*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?	
?
D__inference_dense_20_layer_call_and_return_conditional_losses_706626

inputs0
matmul_readvariableop_resource:#-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:#*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?

?
D__inference_dense_18_layer_call_and_return_conditional_losses_706587

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_dense_21_layer_call_fn_706635

inputs
unknown:
	unknown_0:
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
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_706358o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?H
?
"__inference__traced_restore_706788
file_prefixF
4assignvariableop_wide_and_deep_model_dense_18_kernel:B
4assignvariableop_1_wide_and_deep_model_dense_18_bias:H
6assignvariableop_2_wide_and_deep_model_dense_19_kernel:B
4assignvariableop_3_wide_and_deep_model_dense_19_bias:H
6assignvariableop_4_wide_and_deep_model_dense_20_kernel:#B
4assignvariableop_5_wide_and_deep_model_dense_20_bias:H
6assignvariableop_6_wide_and_deep_model_dense_21_kernel:B
4assignvariableop_7_wide_and_deep_model_dense_21_bias:%
assignvariableop_8_sgd_iter:	 &
assignvariableop_9_sgd_decay: /
%assignvariableop_10_sgd_learning_rate: *
 assignvariableop_11_sgd_momentum: %
assignvariableop_12_total_2: %
assignvariableop_13_count_2: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 
identity_19??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp4assignvariableop_wide_and_deep_model_dense_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp4assignvariableop_1_wide_and_deep_model_dense_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp6assignvariableop_2_wide_and_deep_model_dense_19_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp4assignvariableop_3_wide_and_deep_model_dense_19_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp6assignvariableop_4_wide_and_deep_model_dense_20_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp4assignvariableop_5_wide_and_deep_model_dense_20_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp6assignvariableop_6_wide_and_deep_model_dense_21_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp4assignvariableop_7_wide_and_deep_model_dense_21_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_sgd_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp%assignvariableop_10_sgd_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp assignvariableop_11_sgd_momentumIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
?
?
)__inference_dense_20_layer_call_fn_706616

inputs
unknown:#
	unknown_0:
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
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_706342o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
?
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706366

inputs
inputs_1!
dense_18_706308:
dense_18_706310:!
dense_19_706325:
dense_19_706327:!
dense_20_706343:#
dense_20_706345:!
dense_21_706359:
dense_21_706361:
identity

identity_1?? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_18_706308dense_18_706310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_706307?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_706325dense_19_706327*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_706324Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2inputs)dense_19/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#?
 dense_20/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0dense_20_706343dense_20_706345*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_706342?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_21_706359dense_21_706361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_706358x
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????z

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706479
input_1
input_2!
dense_18_706455:
dense_18_706457:!
dense_19_706460:
dense_19_706462:!
dense_20_706467:#
dense_20_706469:!
dense_21_706472:
dense_21_706474:
identity

identity_1?? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_18_706455dense_18_706457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_706307?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_706460dense_19_706462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_706324Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2input_1)dense_19/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#?
 dense_20/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0dense_20_706467dense_20_706469*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_706342?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_21_706472dense_21_706474*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_706358x
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????z

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
4__inference_wide_and_deep_model_layer_call_fn_706533
inputs_0
inputs_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706366o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
D__inference_dense_18_layer_call_and_return_conditional_losses_706307

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
4__inference_wide_and_deep_model_layer_call_fn_706387
input_1
input_2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706366o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
D__inference_dense_19_layer_call_and_return_conditional_losses_706607

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
__inference__traced_save_706724
file_prefixB
>savev2_wide_and_deep_model_dense_18_kernel_read_readvariableop@
<savev2_wide_and_deep_model_dense_18_bias_read_readvariableopB
>savev2_wide_and_deep_model_dense_19_kernel_read_readvariableop@
<savev2_wide_and_deep_model_dense_19_bias_read_readvariableopB
>savev2_wide_and_deep_model_dense_20_kernel_read_readvariableop@
<savev2_wide_and_deep_model_dense_20_bias_read_readvariableopB
>savev2_wide_and_deep_model_dense_21_kernel_read_readvariableop@
<savev2_wide_and_deep_model_dense_21_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_wide_and_deep_model_dense_18_kernel_read_readvariableop<savev2_wide_and_deep_model_dense_18_bias_read_readvariableop>savev2_wide_and_deep_model_dense_19_kernel_read_readvariableop<savev2_wide_and_deep_model_dense_19_bias_read_readvariableop>savev2_wide_and_deep_model_dense_20_kernel_read_readvariableop<savev2_wide_and_deep_model_dense_20_bias_read_readvariableop>savev2_wide_and_deep_model_dense_21_kernel_read_readvariableop<savev2_wide_and_deep_model_dense_21_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*k
_input_shapesZ
X: :::::#:::: : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:#: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?	
?
D__inference_dense_21_layer_call_and_return_conditional_losses_706358

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
!__inference__wrapped_model_706287
input_1
input_2M
;wide_and_deep_model_dense_18_matmul_readvariableop_resource:J
<wide_and_deep_model_dense_18_biasadd_readvariableop_resource:M
;wide_and_deep_model_dense_19_matmul_readvariableop_resource:J
<wide_and_deep_model_dense_19_biasadd_readvariableop_resource:M
;wide_and_deep_model_dense_20_matmul_readvariableop_resource:#J
<wide_and_deep_model_dense_20_biasadd_readvariableop_resource:M
;wide_and_deep_model_dense_21_matmul_readvariableop_resource:J
<wide_and_deep_model_dense_21_biasadd_readvariableop_resource:
identity

identity_1??3wide_and_deep_model/dense_18/BiasAdd/ReadVariableOp?2wide_and_deep_model/dense_18/MatMul/ReadVariableOp?3wide_and_deep_model/dense_19/BiasAdd/ReadVariableOp?2wide_and_deep_model/dense_19/MatMul/ReadVariableOp?3wide_and_deep_model/dense_20/BiasAdd/ReadVariableOp?2wide_and_deep_model/dense_20/MatMul/ReadVariableOp?3wide_and_deep_model/dense_21/BiasAdd/ReadVariableOp?2wide_and_deep_model/dense_21/MatMul/ReadVariableOp?
2wide_and_deep_model/dense_18/MatMul/ReadVariableOpReadVariableOp;wide_and_deep_model_dense_18_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#wide_and_deep_model/dense_18/MatMulMatMulinput_2:wide_and_deep_model/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
3wide_and_deep_model/dense_18/BiasAdd/ReadVariableOpReadVariableOp<wide_and_deep_model_dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$wide_and_deep_model/dense_18/BiasAddBiasAdd-wide_and_deep_model/dense_18/MatMul:product:0;wide_and_deep_model/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!wide_and_deep_model/dense_18/ReluRelu-wide_and_deep_model/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
2wide_and_deep_model/dense_19/MatMul/ReadVariableOpReadVariableOp;wide_and_deep_model_dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#wide_and_deep_model/dense_19/MatMulMatMul/wide_and_deep_model/dense_18/Relu:activations:0:wide_and_deep_model/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
3wide_and_deep_model/dense_19/BiasAdd/ReadVariableOpReadVariableOp<wide_and_deep_model_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$wide_and_deep_model/dense_19/BiasAddBiasAdd-wide_and_deep_model/dense_19/MatMul:product:0;wide_and_deep_model/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!wide_and_deep_model/dense_19/ReluRelu-wide_and_deep_model/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m
+wide_and_deep_model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
&wide_and_deep_model/concatenate/concatConcatV2input_1/wide_and_deep_model/dense_19/Relu:activations:04wide_and_deep_model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#?
2wide_and_deep_model/dense_20/MatMul/ReadVariableOpReadVariableOp;wide_and_deep_model_dense_20_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0?
#wide_and_deep_model/dense_20/MatMulMatMul/wide_and_deep_model/concatenate/concat:output:0:wide_and_deep_model/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
3wide_and_deep_model/dense_20/BiasAdd/ReadVariableOpReadVariableOp<wide_and_deep_model_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$wide_and_deep_model/dense_20/BiasAddBiasAdd-wide_and_deep_model/dense_20/MatMul:product:0;wide_and_deep_model/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
2wide_and_deep_model/dense_21/MatMul/ReadVariableOpReadVariableOp;wide_and_deep_model_dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
#wide_and_deep_model/dense_21/MatMulMatMul/wide_and_deep_model/dense_19/Relu:activations:0:wide_and_deep_model/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
3wide_and_deep_model/dense_21/BiasAdd/ReadVariableOpReadVariableOp<wide_and_deep_model_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$wide_and_deep_model/dense_21/BiasAddBiasAdd-wide_and_deep_model/dense_21/MatMul:product:0;wide_and_deep_model/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
IdentityIdentity-wide_and_deep_model/dense_20/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????~

Identity_1Identity-wide_and_deep_model/dense_21/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp4^wide_and_deep_model/dense_18/BiasAdd/ReadVariableOp3^wide_and_deep_model/dense_18/MatMul/ReadVariableOp4^wide_and_deep_model/dense_19/BiasAdd/ReadVariableOp3^wide_and_deep_model/dense_19/MatMul/ReadVariableOp4^wide_and_deep_model/dense_20/BiasAdd/ReadVariableOp3^wide_and_deep_model/dense_20/MatMul/ReadVariableOp4^wide_and_deep_model/dense_21/BiasAdd/ReadVariableOp3^wide_and_deep_model/dense_21/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2j
3wide_and_deep_model/dense_18/BiasAdd/ReadVariableOp3wide_and_deep_model/dense_18/BiasAdd/ReadVariableOp2h
2wide_and_deep_model/dense_18/MatMul/ReadVariableOp2wide_and_deep_model/dense_18/MatMul/ReadVariableOp2j
3wide_and_deep_model/dense_19/BiasAdd/ReadVariableOp3wide_and_deep_model/dense_19/BiasAdd/ReadVariableOp2h
2wide_and_deep_model/dense_19/MatMul/ReadVariableOp2wide_and_deep_model/dense_19/MatMul/ReadVariableOp2j
3wide_and_deep_model/dense_20/BiasAdd/ReadVariableOp3wide_and_deep_model/dense_20/BiasAdd/ReadVariableOp2h
2wide_and_deep_model/dense_20/MatMul/ReadVariableOp2wide_and_deep_model/dense_20/MatMul/ReadVariableOp2j
3wide_and_deep_model/dense_21/BiasAdd/ReadVariableOp3wide_and_deep_model/dense_21/BiasAdd/ReadVariableOp2h
2wide_and_deep_model/dense_21/MatMul/ReadVariableOp2wide_and_deep_model/dense_21/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
$__inference_signature_wrapper_706509
input_1
input_2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_706287o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
)__inference_dense_18_layer_call_fn_706576

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_706307o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_21_layer_call_and_return_conditional_losses_706645

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
;
input_20
serving_default_input_2:0?????????<
output_10
StatefulPartitionedCall:0?????????<
output_20
StatefulPartitionedCall:1?????????tensorflow/serving/predict:?o
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
hidden1
	hidden2

main_output

aux_output
	optimizer
loss

signatures"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
trace_0
trace_12?
4__inference_wide_and_deep_model_layer_call_fn_706387
4__inference_wide_and_deep_model_layer_call_fn_706533?
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
 ztrace_0ztrace_1
?
trace_0
trace_12?
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706567
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706479?
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
 ztrace_0ztrace_1
?B?
!__inference__wrapped_model_706287input_1input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
I
8iter
	9decay
:learning_rate
;momentum"
	optimizer
 "
trackable_list_wrapper
,
<serving_default"
signature_map
5:32#wide_and_deep_model/dense_18/kernel
/:-2!wide_and_deep_model/dense_18/bias
5:32#wide_and_deep_model/dense_19/kernel
/:-2!wide_and_deep_model/dense_19/bias
5:3#2#wide_and_deep_model/dense_20/kernel
/:-2!wide_and_deep_model/dense_20/bias
5:32#wide_and_deep_model/dense_21/kernel
/:-2!wide_and_deep_model/dense_21/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
4__inference_wide_and_deep_model_layer_call_fn_706387input_1input_2"?
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
?B?
4__inference_wide_and_deep_model_layer_call_fn_706533inputs/0inputs/1"?
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
?B?
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706567inputs/0inputs/1"?
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
?B?
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706479input_1input_2"?
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
?
Etrace_02?
)__inference_dense_18_layer_call_fn_706576?
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
 zEtrace_0
?
Ftrace_02?
D__inference_dense_18_layer_call_and_return_conditional_losses_706587?
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
 zFtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?
Ltrace_02?
)__inference_dense_19_layer_call_fn_706596?
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
 zLtrace_0
?
Mtrace_02?
D__inference_dense_19_layer_call_and_return_conditional_losses_706607?
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
 zMtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
?
Strace_02?
)__inference_dense_20_layer_call_fn_706616?
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
 zStrace_0
?
Ttrace_02?
D__inference_dense_20_layer_call_and_return_conditional_losses_706626?
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
 zTtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
?
Ztrace_02?
)__inference_dense_21_layer_call_fn_706635?
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
 zZtrace_0
?
[trace_02?
D__inference_dense_21_layer_call_and_return_conditional_losses_706645?
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
 z[trace_0
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
?B?
$__inference_signature_wrapper_706509input_1input_2"?
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
 
N
\	variables
]	keras_api
	^total
	_count"
_tf_keras_metric
N
`	variables
a	keras_api
	btotal
	ccount"
_tf_keras_metric
N
d	variables
e	keras_api
	ftotal
	gcount"
_tf_keras_metric
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
?B?
)__inference_dense_18_layer_call_fn_706576inputs"?
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
?B?
D__inference_dense_18_layer_call_and_return_conditional_losses_706587inputs"?
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
?B?
)__inference_dense_19_layer_call_fn_706596inputs"?
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
?B?
D__inference_dense_19_layer_call_and_return_conditional_losses_706607inputs"?
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
?B?
)__inference_dense_20_layer_call_fn_706616inputs"?
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
?B?
D__inference_dense_20_layer_call_and_return_conditional_losses_706626inputs"?
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
?B?
)__inference_dense_21_layer_call_fn_706635inputs"?
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
?B?
D__inference_dense_21_layer_call_and_return_conditional_losses_706645inputs"?
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
.
^0
_1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
.
b0
c1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
:  (2total
:  (2count?
!__inference__wrapped_model_706287?X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "c?`
.
output_1"?
output_1?????????
.
output_2"?
output_2??????????
D__inference_dense_18_layer_call_and_return_conditional_losses_706587\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_18_layer_call_fn_706576O/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_dense_19_layer_call_and_return_conditional_losses_706607\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_19_layer_call_fn_706596O/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_dense_20_layer_call_and_return_conditional_losses_706626\/?,
%?"
 ?
inputs?????????#
? "%?"
?
0?????????
? |
)__inference_dense_20_layer_call_fn_706616O/?,
%?"
 ?
inputs?????????#
? "???????????
D__inference_dense_21_layer_call_and_return_conditional_losses_706645\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_21_layer_call_fn_706635O/?,
%?"
 ?
inputs?????????
? "???????????
$__inference_signature_wrapper_706509?i?f
? 
_?\
,
input_1!?
input_1?????????
,
input_2!?
input_2?????????"c?`
.
output_1"?
output_1?????????
.
output_2"?
output_2??????????
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706479?X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
O__inference_wide_and_deep_model_layer_call_and_return_conditional_losses_706567?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
4__inference_wide_and_deep_model_layer_call_fn_706387?X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "=?:
?
0?????????
?
1??????????
4__inference_wide_and_deep_model_layer_call_fn_706533?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "=?:
?
0?????????
?
1?????????