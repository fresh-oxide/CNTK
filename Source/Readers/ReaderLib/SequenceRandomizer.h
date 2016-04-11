//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>

#include "Transformer.h"
#include "DataDeserializer.h"
#include "ChunkRandomizer.h"
#include <deque>

namespace Microsoft { namespace MSR { namespace CNTK {

// Randomized sequence description.
struct RandomizedSequenceDescription
{
    // Sequence id.
    size_t m_id;
    // Number of samples in sequence.
    size_t m_numberOfSamples;
    // Randomized chunk this sequence belongs to.
    const RandomizedChunk* m_chunk;
};

// Class that given randomized chunks, randomizes sequence descriptions in a window of chunks.
// TODO: This code is still based on the old behavior, so that all current tests pass.
// TODO: Can be simplified if we only randomized sequences forward.
class SequenceRandomizer
{
public:
    SequenceRandomizer(
        IDataDeserializerPtr deserializer,
        ChunkRandomizerPtr chunkRandomizer);

    // Resets current sequence sweep according to the seed.
    void Reset(size_t seed);

    // Sets current sequence cursor given the sample offset in a sweep.
    // If the sample offset point in the middle of sequence, the cursor is moved to the sequence end,
    // and a new sample offset is returned that points to the end of the sequence.
    size_t Seek(size_t sweepSampleOffset, size_t sweep);

    // Gets next sequence descriptions.
    std::vector<RandomizedSequenceDescription> GetNextSequenceDescriptions(size_t sampleCount);

    // Gets current randomized chunk window.
    const std::deque<RandomizedChunk>& GetChunkWindow() const
    {
        return m_chunkWindow;
    }

    void ReleaseChunks();

private:
    DISABLE_COPY_AND_MOVE(SequenceRandomizer);

    void RandomizeNextChunkIfNeeded();

    // Validates if sequence description is valid for the current position.
    bool IsValidForPosition(size_t targetPosition, const RandomizedSequenceDescription& seqDesc) const;

    // Gets randomized chunk index by the sequence position inside the sweep.
    size_t GetChunkIndexForSequencePosition(size_t sequencePosition) const;

    // Gets randomized sequence description by the sample offset in the sweep.
    RandomizedSequenceDescription& GetRandomizedSequenceDescriptionBySequenceId(size_t sequenceId);

    // Adds randomized sequences to the window.
    void AddRandomizedSequencesForChunk(size_t chunkIndex);

    // TODO add a comment, rename
    void MoveChunkCursor();

private:

    IDataDeserializerPtr m_deserializer;

    // Used only as a buffer to get sequence descriptions without memory reallocation.
    std::vector<SequenceDescription> m_bufferOriginalSequences;

    // Randomized chunks.
    const std::vector<RandomizedChunk>& m_randomizedChunks;

    //
    // We randomize sequences in a rolling window over the randomized chunks.
    // This window is organized into chunks, where the chunk indices and number of sequences
    // are equal to the randomized chunks. The number of samples in each randomized chunk, however,
    // may vary due to sequences being changed.
    //
    // The rolling window can be divided into three parts. The first part is fully randomized, and
    // has sequences at their final position (wrt. the randomization for the sweep). Only sequences
    // from this part are returned to the caller.
    // The second and third part correspond to sequences that are being randomized, i.e., within
    // which sequences may still change their position. The randomization cursor, which is located
    // at the boundary between part 2 and 3, indicates where to continue randomization by
    // swapping sequences forward or backward subject to randomziation window conditions.
    //
    //                              all chunks:
    //                          m_randomizedChunks[]
    //  ----------+------------+---------------+---------------------+-------------
    //            |               loaded chunks:                     |
    //            |      m_chunkWindow[], m_sequenceWindow[]         |
    //   unloaded +------------+------------------+------------------+ chunks to be
    //    chunks  | randomized | in randomization | in randomization |   loaded
    //            |            | (back window)    | (forward window) |
    //  ----------+------------+------------------+------------------+-------------
    //            |     ^      |                  |                  |
    //            |     |      |                  |                  | m_chunkWindowEnd
    //            |     |      |                  |
    //            |     |      |                  | m_randomizationCursor
    //            |     |      |
    //            |     |      | m_randomizedWindowEnd
    //            |     |
    //            |     | m_currentChunkCursor
    //            |
    //            | m_chunkWindowBegin
    //
    //

    // A rolling windows of randomized chunks.
    // Which chunk to load is decided by the BlockRandomizer (i.e. decimation based on chunk).
    std::deque<RandomizedChunk> m_chunkWindow;

    // A rolling window of randomized sequences for the chunks.
    std::deque<std::vector<RandomizedSequenceDescription>> m_sequenceWindow;

    struct ChunkInfo
    {
        size_t start;
        size_t numberOfSamples;
    };

    // A rolling window of sample start positions and length for chunks which had their
    // sequenced randomized.
    std::deque<ChunkInfo> m_randomizedChunkInfo;

    // Index of the first chunk in the window (inclusive).
    size_t m_chunkWindowBegin;

    // Indices of chunk, sequence, and sample from which to return data to caller.
    size_t m_currentChunkCursor;
    size_t m_currentSequenceCursor;
    size_t m_currentSampleCursor;

    // Index of the last fully randomized chunk in the window (exclusive).
    size_t m_randomizedWindowEnd;

    // Index of the chunk in the window where to continue randomizing sequences.
    size_t m_randomizationCursor;

    // Index of the last chunk in the window (exclusive).
    size_t m_chunkWindowEnd;
};

typedef std::shared_ptr<SequenceRandomizer> SequenceRandomizerPtr;
}}}
