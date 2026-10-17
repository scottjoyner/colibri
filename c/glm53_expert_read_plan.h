/* Native GLM5.3 six-tensor IO planner (pure, bounded, no disk or allocations).
 * Sorts physical shard extents and maps every logical tensor into a packed slab.
 * Header-only for the GLM engine and its standalone correctness harness.
 * Experimental runtime opt-in only. */
#ifndef GLM53_EXPERT_READ_PLAN_H
#define GLM53_EXPERT_READ_PLAN_H
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

#define G53_IO_PIECES 6
typedef struct {
    int fd;
    int64_t off, len;
    size_t dest;
} G53IOrun;
typedef struct {
    G53IOrun run[G53_IO_PIECES];
    size_t piece_dest[G53_IO_PIECES];
    int nruns;
    size_t total_bytes;
} G53IOplan;

/* Returns 1 only when all six input tensors are nonempty, nonoverlapping,
 * bounded by INT64_MAX and fit the caller's *existing* destination slab.
 * No gap overfetch or cross-file joining. Physical ordering is arbitrary;
 * the six logical tensor indices retain their original identity. */
static int g53_io_plan(const int fd[G53_IO_PIECES],
                       const int64_t off[G53_IO_PIECES],
                       const int64_t len[G53_IO_PIECES],
                       size_t capacity, G53IOplan *out) {
    if (!fd || !off || !len || !out || !capacity) return 0;
    int order[G53_IO_PIECES] = {0,1,2,3,4,5};
    for (int p = 0; p < G53_IO_PIECES; p++) {
        if (fd[p] < 0 || off[p] < 0 || len[p] <= 0 ||
            len[p] > INT64_MAX-off[p] || (uint64_t)len[p] > SIZE_MAX)
            return 0;
    }
    for (int i = 1; i < G53_IO_PIECES; i++) {
        int p=order[i], j=i;
        while (j>0 && (fd[order[j-1]]>fd[p] ||
              (fd[order[j-1]]==fd[p] && off[order[j-1]]>off[p]))) {
            order[j]=order[j-1];j--;
        }
        order[j]=p;
    }
    G53IOplan result = {0};
    for (int i = 0; i < G53_IO_PIECES; i++) {
        int p=order[i];
        size_t n=(size_t)len[p];
        if (n>capacity-result.total_bytes) return 0;
        if (result.nruns) {
            G53IOrun *prev=&result.run[result.nruns-1];
            int64_t end=prev->off+prev->len;
            if (prev->fd==fd[p] && off[p]<end) return 0;
            if (prev->fd==fd[p] && off[p]==end) {
                if (len[p]>INT64_MAX-prev->len) return 0;
                prev->len+=len[p];
            } else {
                result.run[result.nruns++] = (G53IOrun){
                    .fd=fd[p],.off=off[p],.len=len[p],
                    .dest=result.total_bytes};
            }
        } else {
            result.run[result.nruns++] = (G53IOrun){
                .fd=fd[p],.off=off[p],.len=len[p],.dest=0};
        }
        result.piece_dest[p]=result.total_bytes;
        result.total_bytes+=n;
    }
    if (result.total_bytes!=capacity || !result.nruns) return 0;
    *out=result;
    return 1;
}
#endif
