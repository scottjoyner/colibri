/* Linux-only bounded preadv scatter helper. Native GLM5.3 opt-in.
 * All iovecs and the initial file interval are checked by caller.
 * Retry EINTR and partial reads without rereading or skipping any tensor.
 * Return 0 (fail closed) on EOF/error; never silently downgrade to mmap.
 */
#ifndef GLM53_EXPERT_PREADV_H
#define GLM53_EXPERT_PREADV_H
#if defined(__linux__)
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>
static int g53_preadv_full(int fd,const struct iovec *original,int n,
                           int64_t file_offset,size_t bytes,unsigned int *calls) {
    if(fd<0 || !original || n<1 || n>6 || file_offset<0 || !bytes ||
       (uint64_t)bytes>INT64_MAX-(uint64_t)file_offset)return 0;
    struct iovec vec[6];
    size_t total=0;
    for(int i=0;i<n;i++){
        if(!original[i].iov_base || !original[i].iov_len ||
           original[i].iov_len>bytes-total)return 0;
        vec[i]=original[i];
        total+=vec[i].iov_len;
    }
    if(total!=bytes)return 0;
    unsigned int reads=0;
    size_t consumed=0;
    int current=0;
    while(consumed<bytes){
        ssize_t got=preadv(fd,vec+current,n-current,
                           (off_t)(file_offset+(int64_t)consumed));
        if(got<0 && errno==EINTR)continue;
        reads++;
        if(got<=0 || (size_t)got>bytes-consumed)return 0;
        size_t done=(size_t)got;
        consumed+=done;
        while(done){
            if(current>=n)return 0;
            size_t piece=vec[current].iov_len;
            if(done>=piece){done-=piece;current++;}
            else{
                vec[current].iov_base=(unsigned char *)vec[current].iov_base+done;
                vec[current].iov_len-=done;
                done=0;
            }
        }
    }
    if(current!=n)return 0;
    if(calls)*calls+=reads;
    return 1;
}
#endif
#endif
