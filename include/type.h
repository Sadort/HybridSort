class SortIndices
{
   private:
     uint64_t* mparr;
   public:
     SortIndices(uint64_t* parr) : mparr(parr) {}
     bool operator()(uint64_t i, uint64_t j) const { return mparr[i]<mparr[j]; }
}
