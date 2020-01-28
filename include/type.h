const unsigned long MASK = 0xFFFF000000000000;

__host__ __device__ bool operator<(const ulong2 &a, const ulong2 &b) {
    if      (a.x < b.x) return true;
    else if (a.x == b.x && (a.y&MASK) <= (b.y&MASK)) return true;
    else return false;
}