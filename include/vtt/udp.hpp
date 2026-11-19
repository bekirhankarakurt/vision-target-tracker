/// @file udp.hpp
/// @brief Minimal non-blocking IPv4 UDP socket (Winsock or POSIX), RAII.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vtt {

/// Non-blocking UDP socket. Construction throws std::runtime_error on failure;
/// send and receive never throw and never block.
class UdpSocket {
public:
    /// Bind to 127.0.0.1:@p port to receive; port 0 binds an ephemeral port (send only).
    explicit UdpSocket(std::uint16_t port = 0);
    ~UdpSocket();

    UdpSocket(const UdpSocket&) = delete;
    UdpSocket& operator=(const UdpSocket&) = delete;
    UdpSocket(UdpSocket&&) = delete;
    UdpSocket& operator=(UdpSocket&&) = delete;

    /// Send one datagram to an IPv4 address such as "127.0.0.1". Returns false on error.
    bool send_to(const std::string& host, std::uint16_t port,
                 const std::vector<std::uint8_t>& bytes) noexcept;

    /// Next pending datagram, or empty when none is waiting.
    [[nodiscard]] std::optional<std::vector<std::uint8_t>> try_receive(std::size_t max_bytes = 2048) noexcept;

private:
    std::uintptr_t handle_;
};

/// Parse "HOST:PORT" (IPv4 dotted host, port 1..65535).
struct Endpoint {
    std::string host;
    std::uint16_t port = 0;
};
[[nodiscard]] std::optional<Endpoint> parse_endpoint(const std::string& text);

}  // namespace vtt
