#include "vtt/udp.hpp"

#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace vtt {
namespace {

#ifdef _WIN32
using NativeSocket = SOCKET;
const NativeSocket kInvalid = INVALID_SOCKET;

/// Winsock is started once for the process and stopped at exit.
struct WinsockSession {
    WinsockSession() {
        WSADATA data{};
        ok = WSAStartup(MAKEWORD(2, 2), &data) == 0;
    }
    ~WinsockSession() {
        if (ok) {
            WSACleanup();
        }
    }
    WinsockSession(const WinsockSession&) = delete;
    WinsockSession& operator=(const WinsockSession&) = delete;
    bool ok = false;
};

bool ensure_winsock() {
    static const WinsockSession session;
    return session.ok;
}

void close_native(NativeSocket s) { closesocket(s); }
bool make_non_blocking(NativeSocket s) {
    u_long mode = 1;
    return ioctlsocket(s, FIONBIO, &mode) == 0;
}
#else
using NativeSocket = int;
const NativeSocket kInvalid = -1;
bool ensure_winsock() { return true; }
void close_native(NativeSocket s) { ::close(s); }
bool make_non_blocking(NativeSocket s) {
    const int flags = fcntl(s, F_GETFL, 0);
    return flags >= 0 && fcntl(s, F_SETFL, flags | O_NONBLOCK) == 0;
}
#endif

NativeSocket native(std::uintptr_t handle) { return static_cast<NativeSocket>(handle); }

sockaddr_in make_address(const std::string& host, std::uint16_t port, bool& ok) {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    ok = inet_pton(AF_INET, host.c_str(), &address.sin_addr) == 1;
    return address;
}

}  // namespace

UdpSocket::UdpSocket(std::uint16_t port) : handle_(static_cast<std::uintptr_t>(kInvalid)) {
    if (!ensure_winsock()) {
        throw std::runtime_error("cannot start Winsock");
    }
    const NativeSocket s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == kInvalid) {
        throw std::runtime_error("cannot create UDP socket");
    }
    handle_ = static_cast<std::uintptr_t>(s);
    bool ok = false;
    const sockaddr_in address = make_address("127.0.0.1", port, ok);
    if (!ok || ::bind(s, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        close_native(s);
        throw std::runtime_error("cannot bind UDP port " + std::to_string(port));
    }
    if (!make_non_blocking(s)) {
        close_native(s);
        throw std::runtime_error("cannot make UDP socket non-blocking");
    }
}

UdpSocket::~UdpSocket() { close_native(native(handle_)); }

bool UdpSocket::send_to(const std::string& host, std::uint16_t port,
                        const std::vector<std::uint8_t>& bytes) noexcept {
    bool ok = false;
    const sockaddr_in address = make_address(host, port, ok);
    if (!ok || bytes.empty()) {
        return false;
    }
    const auto sent = ::sendto(native(handle_), reinterpret_cast<const char*>(bytes.data()),
#ifdef _WIN32
                               static_cast<int>(bytes.size()),
#else
                               bytes.size(),
#endif
                               0, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
    return sent >= 0 && static_cast<std::size_t>(sent) == bytes.size();
}

std::optional<std::vector<std::uint8_t>> UdpSocket::try_receive(std::size_t max_bytes) noexcept {
    try {
        std::vector<std::uint8_t> buffer(max_bytes);
        const auto got = ::recvfrom(native(handle_), reinterpret_cast<char*>(buffer.data()),
#ifdef _WIN32
                                    static_cast<int>(buffer.size()),
#else
                                    buffer.size(),
#endif
                                    0, nullptr, nullptr);
        if (got <= 0) {
            return std::nullopt;  // Nothing waiting (would block) or an error.
        }
        buffer.resize(static_cast<std::size_t>(got));
        return buffer;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<Endpoint> parse_endpoint(const std::string& text) {
    const std::size_t colon = text.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= text.size()) {
        return std::nullopt;
    }
    const std::string host = text.substr(0, colon);
    const std::string port_text = text.substr(colon + 1);
    char* end = nullptr;
    const long port = std::strtol(port_text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || port < 1 || port > 65535) {
        return std::nullopt;
    }
    bool ok = false;
    static_cast<void>(make_address(host, 0, ok));
    if (!ok) {
        return std::nullopt;
    }
    return Endpoint{host, static_cast<std::uint16_t>(port)};
}

}  // namespace vtt
