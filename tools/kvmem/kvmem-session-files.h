#pragma once

#include "kvmem/snapshot.hpp"
#include "kvmem/spill_io.hpp"
#include <filesystem>
#include <map>
#include <string>

using kvmem_session_corrupt = kvmem::SnapshotCorrupt;

// State attachments share the native KV spill file, quota and lifetime.
// Unpublished writes retain their extent locally and release it on failure.
class kvmem_session_files {
public:
    kvmem_session_files(const std::filesystem::path & root, std::shared_ptr<kvmem::SpillFile> shared)
        : dir_(root), shared_(std::move(shared)) {
        if (!shared_) throw std::invalid_argument("session attachments require the shared KV spill file");
    }
    kvmem_session_files(const kvmem_session_files &) = delete;
    kvmem_session_files & operator=(const kvmem_session_files &) = delete;
    uint64_t bytes() const { return shared_->charged_bytes(); }
    uint64_t limit() const { return shared_->capacity_bytes(); }
    uint64_t available() const { return std::filesystem::space(dir_).available; }
    const std::filesystem::path & directory() const { return dir_; }

    // Optional deterministic fault injection for the portable transfer tests.
    std::function<void(const char *, int, uint32_t)> fault;
    const kvmem::SpillExtent & extent(int id, uint32_t chunk = 0) const { return files_.at({id, chunk}).extent; }
    bool contains(int id) const {
        auto it = files_.lower_bound({id, 0}); return it != files_.end() && it->first.first == id;
    }
    bool contains_chunk(int id, uint32_t chunk) const { return files_.count({id, chunk}) != 0; }
    bool ready(int id, uint32_t chunk) const {
        return contains_chunk(id, chunk);
    }
    uint64_t chunk_bytes(int id, uint32_t chunk) const {
        auto it = files_.find({id, chunk}); return it == files_.end() ? 0 : it->second.bytes;
    }
    uint64_t session_bytes(int id) const {
        uint64_t n = 0;
        for (auto it = files_.lower_bound({id, 0}); it != files_.end() && it->first.first == id; ++it) n += it->second.bytes;
        return n;
    }
    bool erase(int id) {
        bool ok = true;
        for (auto it = files_.lower_bound({id, 0}); it != files_.end() && it->first.first == id;) {
            const auto chunk = it++->first.second;
            if (!erase_chunk(id, chunk)) ok = false;
        }
        return ok;
    }
    bool erase_chunk(int id, uint32_t chunk) {
        const auto it = files_.find({id, chunk});
        if (it == files_.end()) return true;
        try { if (fault) fault("erase", id, chunk); } catch (...) { return false; }
        files_.erase(it);
        return true;
    }
    template<class Write> void save(int id, uint64_t payload_bytes, const Write & write) {
        save_chunk(id, 0, payload_bytes, write);
    }
    template<class Write> void save_chunk(int id, uint32_t chunk, uint64_t payload_bytes, const Write & write) {
        if (contains_chunk(id, chunk) || payload_bytes > UINT64_MAX - 8 || payload_bytes + 8 > limit() - bytes())
            throw std::runtime_error("session disk quota exceeded");
        const auto total = payload_bytes + 8;
        if (available() < total)
            throw std::runtime_error("insufficient free space for session snapshot");
        auto extent = shared_->reserve(total);
        if (fault) fault("write", id, chunk);
        uint64_t offset = 0;
        kvmem::SnapshotWriter out([&](const void * p, size_t n) {
            if (fault) fault("write_data", id, chunk);
            extent.write(offset, p, n); offset += n;
        });
        write(out);
        if (out.bytes() != payload_bytes) throw std::runtime_error("session changed during snapshot");
        const auto hash = out.hash(); extent.write(offset, &hash, sizeof(hash));
        if (fault) fault("publish", id, chunk);
        files_.emplace(std::make_pair(id, chunk), record{total, std::move(extent)});
    }
    template<class Read> void load(int id, const Read & read) const {
        load_chunk(id, 0, read);
    }
    template<class Read> void load_chunk(int id, uint32_t chunk, const Read & read) const {
        const auto & record = files_.at({id, chunk});
        if (fault) fault("read", id, chunk);
        if (record.bytes < 8 || record.extent.size() != record.bytes)
            throw kvmem_session_corrupt("invalid shared session extent");
        uint64_t offset = 0;
        kvmem::SnapshotReader in([&](void * p, size_t n) {
            if (fault) fault("read_data", id, chunk);
            record.extent.read(offset, p, n); offset += n;
        }, record.bytes - 8);
        read(in);
        uint64_t hash = 0; record.extent.read(offset, &hash, sizeof(hash));
        if (in.remaining() || hash != in.hash()) throw kvmem_session_corrupt("session snapshot checksum mismatch");
        if (fault) fault("validated", id, chunk);
    }

private:
    struct record { uint64_t bytes; kvmem::SpillExtent extent; };
    std::filesystem::path dir_;
    std::map<std::pair<int, uint32_t>, record> files_;
    std::shared_ptr<kvmem::SpillFile> shared_;
};
