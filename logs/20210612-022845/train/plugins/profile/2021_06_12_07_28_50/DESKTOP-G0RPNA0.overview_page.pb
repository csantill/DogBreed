?!  *??(\?c?@X9?ȶ[?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?Wt?5? @!8?ʤ|DR@)F&??H? @19?8???Q@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap[0]::TFRecord???e???!?8j%3@)???e???1?8j%3@:Advanced file read2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2ds?<G???!g")Vxc??)ds?<G???1g")Vxc??:Preprocessing2?
ZIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::ShuffleF???jH??!???pAi??)?i??_=??1l??n?A??:Preprocessing2?
sIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2?cw????! ?r?
??)?cw????1 ?r?
??:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4X<?H?ۚ?!??l????)X<?H?ۚ?1??l????:Preprocessing2?
QIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2j?TQ?ʚ?!%??B????)j?TQ?ʚ?1%??B????:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle?^?D???!??o?8???)?6?xͫ??1[CB?????:Preprocessing2?
dIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch???uS??!D0?N??)???uS??1D0?N??:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinalityJ??4*p??!kfgwm??)<? ???1?b
???:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap?I)?????!?????3@)??1%???1???yk??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?6qr?C??!aG	?T???)?6qr?C??1aG	?T???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism@???൛?!???????)7?C???1?b?v??:Preprocessing2F
Iterator::Model???߃מ?!P??Ӕ??)????i?1???V????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q?I?
?S@"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?76.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.JDESKTOP-G0RPNA0: Failed to load libcupti (is it installed and accessible?)