  *?????+?@ObX?C?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??5&?@!T??d?7L@)ŏ1w-?@1	????K@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[2]::FlatMap[0]::TFRecord}гY??@!?σ??B@)}гY??@1?σ??B@:Advanced file read2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4?O??e??!??6?@)?O??e??1??6?@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2G6u??!?Xa????)G6u??1?Xa????:Preprocessing2?
ZIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle&p?n???!
? S?w??)?Oq??1+n?c???:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[2]::FlatMapd???H?@!CKw??B@)й?????1?????:Preprocessing2?
dIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch?d?????!??Sk??)?d?????1??Sk??:Preprocessing2?
QIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2wg????!???W???)wg????1???W???:Preprocessing2?
sIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2???????!?0R?.???)???????1?0R?.???:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::ParallelMapV2::Shuffle::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinalityѓ2????!	iJ<?R
@)	?Į????1?b?X=%??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchV*?????!8??d??)V*?????18??d??:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle??/?^|??!?R$+??)D??????1p??Y??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?O??0{??!jP"?????)B?Ēr???1?0?????:Preprocessing2F
Iterator::Model??2SZ??!B?!?????)?b?J!p?1???'????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.