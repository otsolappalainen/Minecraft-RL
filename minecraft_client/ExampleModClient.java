package com.example;

// Minecraft imports
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;  
import net.minecraft.client.option.KeyBinding;
import net.minecraft.entity.Entity;
import net.minecraft.entity.ItemEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.effect.StatusEffectInstance;
import net.minecraft.entity.effect.StatusEffects;
import net.minecraft.entity.mob.Monster;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.item.Items;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Direction;
import net.minecraft.world.World;

// WebSocket imports
import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;

// Gson imports
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

// Java imports
import java.lang.reflect.Type;
import java.net.InetSocketAddress;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class ExampleModClient implements ClientModInitializer {
    private final int PORT;

    public ExampleModClient() {
        int instanceId = Integer.parseInt(System.getProperty("instance.id", "0"));
        this.PORT = 8080 + instanceId;
    }

    private WebSocketServer server;
    private MinecraftClient client;
    private Gson gson;

    private double initialX = Double.NaN;
    private double initialZ = Double.NaN;

    private static final int ACTION_DURATION_SHORT = 55;
    private static final int ACTION_DURATION_MEDIUM = 55;
    private static final int EXTENDED_ACTION_TIME = 55;

    private static final float HEAL_CHANCE = 0.04f;
    private static final Random healRandom = new Random();

    private ExecutorService actionExecutor = Executors.newSingleThreadExecutor();
    private static final int STATE_SEND_DELAY = 60;
    private final AtomicBoolean isSneaking = new AtomicBoolean(false);
    private volatile boolean isConnected = false;
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
    private final Map<KeyBinding, ScheduledFuture<?>> keyReleaseTasks = new ConcurrentHashMap<>();
    private ScheduledFuture<?> stateSendTask = null;
    private static final boolean DEBUG_MODE = false;


    private static volatile int MIN_Y_LEVEL = 200;
    private static volatile int MAX_Y_LEVEL = 50;

    private static final ItemStack[] SPAWN_LOADOUT = new ItemStack[] {
        new ItemStack(Items.DIAMOND_PICKAXE, 1),
        new ItemStack(Items.DIAMOND_SWORD, 1),
        new ItemStack(Items.COOKED_BEEF, 20)
    };

    private int actionsSinceLastJump = 0;
    private static final int ACTIONS_REQUIRED_BETWEEN_JUMPS = 20;

    private Entity lastHitMob = null;
    private boolean mobWasHit = false;
    private static final Type ACTION_MAP_TYPE = new TypeToken<Map<String, Object>>(){}.getType();

    @Override
    public void onInitializeClient() {
        client = MinecraftClient.getInstance();
        gson = new Gson();
        cleanupDroppedItems();
        hideUI();
        startWebSocketServer();
    }

    private void startWebSocketServer() {
        server = new WebSocketServer(new InetSocketAddress(PORT)) {
            @Override
            public void onOpen(WebSocket conn, ClientHandshake handshake) {
                isConnected = true;
            }

            @Override
            public void onClose(WebSocket conn, int code, String reason, boolean remote) {
                isConnected = false;
            }

            @Override
            public void onMessage(WebSocket conn, String message) {
                handleAction(conn, message);
            }

            @Override
            public void onError(WebSocket conn, Exception ex) {
                if (DEBUG_MODE) {
                    ex.printStackTrace();
                }
            }

            @Override
            public void onStart() {}
        };

        try {
            server.start();
        } catch (Exception e) {
            System.err.println("[Warning] Failed to start WebSocket server: " + e.getMessage());
        }
    }

    private synchronized void handleAction(WebSocket conn, String message) {
        if (conn == null) {
            System.err.println("[Warning] handleAction called with null connection.");
            return;
        }

        Map<String, Object> action;
        String actionType;
        try {
            action = gson.fromJson(message, ACTION_MAP_TYPE);
            if (action == null) {
                System.err.println("[Warning] Received null or invalid action JSON.");
                sendPlayerState(conn);
                return;
            }
            Object act = action.get("action");
            if (!(act instanceof String)) {
                System.err.println("[Warning] 'action' field is missing or not a string.");
                sendPlayerState(conn);
                return;
            }
            actionType = (String) act;
        } catch (Exception e) {
            System.err.println("[Warning] Failed to parse action: " + e.getMessage());
            sendPlayerState(conn);
            return;
        }

        ClientPlayerEntity player = (client != null) ? client.player : null;
        if (player == null || client == null) {
            System.err.println("[Warning] Player entity or client is null. Sending fallback state.");
            sendPlayerState(conn);
            return;
        }

        try {
            releaseAttackKey();

            if (actionType.startsWith("move_")) {
                cancelOngoingActionsExcept(Arrays.asList(client.options.sneakKey, client.options.attackKey));
            } else if (!actionType.equals("attack") && !actionType.equals("sneak")) {
                cancelOngoingActionsExcept(Arrays.asList(client.options.sneakKey, client.options.attackKey));
            }

            if (actionType.startsWith("reset")) {
                int resetType = Integer.parseInt(actionType.split(" ")[1]);
                handleReset(player, resetType);
                sendPlayerState(conn);

            } else if (actionType.startsWith("spawnrate ")) {
                try {
                    String[] parts = actionType.split(" ");
                    if (parts.length != 10) {
                        System.err.println("[Warning] Invalid spawnrate parameters");
                        sendPlayerState(conn);
                        return;
                    }
                    
                    int x = Integer.parseInt(parts[1]);
                    int y = Integer.parseInt(parts[2]);
                    int z = Integer.parseInt(parts[3]);
                    int maxNearby = Integer.parseInt(parts[4]);
                    int reqRange = Integer.parseInt(parts[5]); 
                    int spawnCount = Integer.parseInt(parts[6]);
                    int spawnRange = Integer.parseInt(parts[7]);
                    int minDelay = Integer.parseInt(parts[8]);
                    int maxDelay = Integer.parseInt(parts[9]);

                    String command = String.format("data merge block %d %d %d {SpawnData:{id:\"minecraft:zombie\"},MinSpawnDelay:%d,MaxSpawnDelay:%d,SpawnCount:%d,MaxNearbyEntities:%d,SpawnRange:%d,RequiredPlayerRange:%d}", 
                        x, y, z, minDelay, maxDelay, spawnCount, maxNearby, spawnRange, reqRange);

                    if (client != null && client.player != null && client.player.networkHandler != null) {
                        client.getNetworkHandler().sendChatCommand(command);
                    }
                    scheduleStateSend(conn);
                } catch (Exception e) {
                    System.err.println("[Warning] Failed to process spawnrate command: " + e.getMessage());
                    scheduleStateSend(conn); 
                }

            } else {
                handleXPAndHealing(player);
                hideUI();
                switch (actionType) {
                    case "move_forward":
                        executeKeyAction(client.options.forwardKey, ACTION_DURATION_MEDIUM, conn);
                        actionsSinceLastJump++;
                        break;
                    case "move_backward":
                        executeKeyAction(client.options.backKey, ACTION_DURATION_MEDIUM, conn);
                        actionsSinceLastJump++;
                        break;
                    case "move_left":
                        executeKeyAction(client.options.leftKey, ACTION_DURATION_MEDIUM, conn);
                        actionsSinceLastJump++;
                        break;
                    case "move_right":
                        executeKeyAction(client.options.rightKey, ACTION_DURATION_MEDIUM, conn);
                        actionsSinceLastJump++;
                        break;
                    case "jump_walk_forward":
                        if (actionsSinceLastJump >= ACTIONS_REQUIRED_BETWEEN_JUMPS) {
                            executeJumpWalkForward(client.options.jumpKey, client.options.forwardKey, conn);
                            actionsSinceLastJump = 0;
                        } else {
                            scheduleStateSend(conn);
                        }
                        break;
                    case "jump":
                        if (actionsSinceLastJump >= ACTIONS_REQUIRED_BETWEEN_JUMPS) {
                            executePlayerAction(() -> {
                                if (player != null) player.jump();
                            }, ACTION_DURATION_MEDIUM, conn);
                            actionsSinceLastJump = 0;
                        } else {
                            scheduleStateSend(conn);
                        }
                        break;
                    case "look_left":
                        executeSmoothAdjustYaw(player, -8, ACTION_DURATION_SHORT, conn);
                        actionsSinceLastJump++;
                        break;
                    case "look_right":
                        executeSmoothAdjustYaw(player, 8, ACTION_DURATION_SHORT, conn);
                        actionsSinceLastJump++;
                        break;
                    case "look_up":
                        executeSmoothAdjustPitch(player, -8, ACTION_DURATION_SHORT, conn);
                        actionsSinceLastJump++;
                        break;
                    case "look_down":
                        executeSmoothAdjustPitch(player, 8, ACTION_DURATION_SHORT, conn);
                        actionsSinceLastJump++;
                        break;
                    case "turn_left":
                        executeSmoothAdjustYaw(player, -10, ACTION_DURATION_MEDIUM, conn);
                        actionsSinceLastJump++;
                        break;
                    case "turn_right":
                        executeSmoothAdjustYaw(player, 10, ACTION_DURATION_MEDIUM, conn);
                        actionsSinceLastJump++;
                        break;
                    case "next_item":
                        executeItemCycle(player, true, conn);
                        actionsSinceLastJump++;
                        break;
                    case "previous_item":
                        executeItemCycle(player, false, conn);
                        actionsSinceLastJump++;
                        break;
                    case "sneak":
                        executeToggleSneak(client.options.sneakKey, conn);
                        actionsSinceLastJump++;
                        break;
                    case "attack":
                        executeAttackAction(client.options.attackKey, conn);
                        actionsSinceLastJump++;
                        break;
                    case "attack 2":
                        executeTimedAttackAction(client.options.attackKey, conn);
                        actionsSinceLastJump++;
                        break;
                    case "use":
                        executeKeyAction(client.options.useKey, ACTION_DURATION_MEDIUM, conn);
                        actionsSinceLastJump++;
                        maintainFullFood(player);
                        break;
                    case "no_op":
                        executeNoOpAction(conn);
                        actionsSinceLastJump++;
                        break;
                    case "monitor":
                        try {
                            int[] coords = getClientWindowCoordinates();
                            Map<String, Object> monitorInfo = new HashMap<>();
                            monitorInfo.put("x", coords[0]);
                            monitorInfo.put("y", coords[1]);
                            monitorInfo.put("width", coords[2]);
                            monitorInfo.put("height", coords[3]);
                            conn.send(gson.toJson(monitorInfo));
                        } catch (Exception e) {
                            System.err.println("[Warning] Error sending monitor info: " + e.getMessage());
                            Map<String, Object> fallback = new HashMap<>();
                            fallback.put("x", 0);
                            fallback.put("y", 0);
                            fallback.put("width", 800);
                            fallback.put("height", 600);
                            conn.send(gson.toJson(fallback));
                        }
                        break;
                    case "tools":
                        executeGiveTool(player, conn);
                        actionsSinceLastJump++;
                        break;


                    default:
                        actionsSinceLastJump++;
                        sendPlayerState(conn);
                        break;
                }
            }
        } catch (Exception e) {
            System.err.println("[Warning] Error executing action: " + e.getMessage());
            sendPlayerState(conn);
        }
    }

    private void releaseAttackKey() {
        if (client == null || client.options == null) return;
        KeyBinding attackKey = client.options.attackKey;
        if (attackKey != null && attackKey.isPressed()) {
            attackKey.setPressed(false);
            KeyBinding.updatePressedStates();
        }
    }

    private void executePlayerAction(Runnable action, long durationMs, WebSocket conn) {
        actionExecutor.submit(() -> {
            try {
                action.run();
            } catch (Exception e) {
                System.err.println("[Warning] Failed to run player action: " + e.getMessage());
            }
        });
        scheduleStateSend(conn);
    }

    private void executeSmoothAdjustYaw(ClientPlayerEntity player, float amount, int durationMs, WebSocket conn) {
        if (player == null) {
            scheduleStateSend(conn);
            return;
        }
        float steps = Math.abs(amount);
        float increment = (steps == 0) ? 0 : amount / steps;

        actionExecutor.submit(() -> {
            try {
                for (int i = 0; i < steps; i++) {
                    if (player == null) break;
                    adjustYaw(player, increment);
                    Thread.sleep(Math.max(1, durationMs / (int) steps));
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } catch (Exception e) {
                System.err.println("[Warning] Error in yaw adjustment: " + e.getMessage());
            }
        });
        scheduleStateSend(conn);
    }

    private void executeSmoothAdjustPitch(ClientPlayerEntity player, float amount, int durationMs, WebSocket conn) {
        if (player == null) {
            scheduleStateSend(conn);
            return;
        }
        float steps = Math.abs(amount);
        float finalAmount = amount;
        float finalPitch = player.getPitch() + amount;
        if (finalPitch > 80) {
            finalAmount = 80 - player.getPitch();
        } else if (finalPitch < -80) {
            finalAmount = -80 - player.getPitch();
        }

        float increment = (steps == 0) ? 0 : finalAmount / steps;

        actionExecutor.submit(() -> {
            try {
                for (int i = 0; i < steps; i++) {
                    if (player == null) break;
                    adjustPitch(player, increment);
                    Thread.sleep(Math.max(1, durationMs / (int) steps));
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } catch (Exception e) {
                System.err.println("[Warning] Error in pitch adjustment: " + e.getMessage());
            }
        });
        scheduleStateSend(conn);
    }

    private void executeToggleSneak(KeyBinding key, WebSocket conn) {
        boolean newState = !isSneaking.getAndSet(!isSneaking.get());
        if (key != null) {
            key.setPressed(newState);
        }
        scheduleStateSend(conn);
    }

    private void executeItemCycle(ClientPlayerEntity player, boolean next, WebSocket conn) {
        if (player == null || player.getInventory() == null) {
            scheduleStateSend(conn);
            return;
        }
        if (next) {
            player.getInventory().selectedSlot = (player.getInventory().selectedSlot + 1) % 9;
        } else {
            player.getInventory().selectedSlot = (player.getInventory().selectedSlot + 8) % 9;
        }
        scheduleStateSend(conn);
    }

    private void maintainFullFood(ClientPlayerEntity player) {
        if (player == null || player.getHungerManager() == null) {
            return;
        }
        player.getHungerManager().setFoodLevel(20);
        player.getHungerManager().setSaturationLevel(20.0f);
    }

    private void handleReset(ClientPlayerEntity player, int resetType) {
        cancelOngoingActionsExcept(null);
        if (stateSendTask != null && !stateSendTask.isDone()) {
            stateSendTask.cancel(false);
            stateSendTask = null;
        }

        if (client != null) {
            client.execute(() -> {
                if (client.player != null && client.player.networkHandler != null) {
                    client.getNetworkHandler().sendChatCommand("time set 13188");
                    client.getNetworkHandler().sendChatCommand("tp @s 31.5 68 30.5");
                }
            });
        }

        actionExecutor.shutdownNow();
        try {
            if (!actionExecutor.awaitTermination(100, TimeUnit.MILLISECONDS)) {
                actionExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            actionExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        actionExecutor = Executors.newSingleThreadExecutor();

        List<KeyBinding> keysToRemove = new ArrayList<>();
        for (Map.Entry<KeyBinding, ScheduledFuture<?>> entry : keyReleaseTasks.entrySet()) {
            ScheduledFuture<?> task = entry.getValue();
            if (task != null && !task.isDone()) {
                task.cancel(false);
            }
            keysToRemove.add(entry.getKey());
        }
        for (KeyBinding k : keysToRemove) {
            k.setPressed(false);
            KeyBinding.updatePressedStates();
            keyReleaseTasks.remove(k);
        }

        if (!player.isAlive()) {
            respawnPlayer();
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        switch (resetType) {
            case 0:
                break;
            case 1:
                try {
                    Thread.sleep(50);
                    double x = player.getX();
                    double y = player.getY();
                    double z = player.getZ();
                    float yaw = player.getYaw();
                    float pitch = player.getPitch();

                    player.refreshPositionAndAngles(x, y, z, yaw, pitch);
                    player.setVelocity(0, 0, 0);
                    player.setOnGround(true);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                break;
            case 2:
                try {
                    Thread.sleep(50);
                    double x = player.getX();
                    double y = player.getY();
                    double z = player.getZ();
                    float yaw = (float) (new Random().nextFloat() * 360 - 180);
                    float pitch = (float) (new Random().nextFloat() * 160 - 80);

                    player.refreshPositionAndAngles(x, y, z, yaw, pitch);
                    player.setVelocity(0, 0, 0);
                    player.setOnGround(true);
                    player.setYaw(yaw);
                    player.setPitch(pitch);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                break;
            default:
                System.err.println("[Warning] Unknown reset type: " + resetType);
        }

        applyNightVision(player);
        maintainFullFood(player);
        isSneaking.set(false);
        if (client != null && client.options != null && client.options.sneakKey != null) {
            client.options.sneakKey.setPressed(false);
        }
        if (client != null && client.options != null && client.options.attackKey != null) {
            client.options.attackKey.setPressed(false);
        }
        KeyBinding.updatePressedStates();
        System.gc();
    }

    private void handlePlayerDeath() {
        if (client == null || client.player == null) return;
        if (!client.player.isAlive()) {
            client.execute(() -> {
                client.player.requestRespawn();
                scheduler.schedule(() -> {
                    if (client.getServer() != null) {
                        client.getServer().execute(() -> {
                            if (client.player != null && client.player.isAlive()) {
                                client.player.getInventory().clear();
                                for (ItemStack item : SPAWN_LOADOUT) {
                                    client.player.getInventory().insertStack(item.copy());
                                }
                                applyNightVision(client.player);
                            }
                        });
                    }
                }, 500, TimeUnit.MILLISECONDS);
            });
        }
    }

    private void respawnPlayer() {
        handlePlayerDeath();
    }

    private void handleXPAndHealing(ClientPlayerEntity player) {
        if (player == null) return;
        if (healRandom.nextFloat() < HEAL_CHANCE) {
            applyNightVision(player);
            float currentHealth = player.getHealth();
            if (currentHealth < player.getMaxHealth()) {
                player.setHealth(Math.min(currentHealth + 1, player.getMaxHealth()));
            }
        }
    }

    private void cancelOngoingActionsExcept(List<KeyBinding> exceptions) {
        List<KeyBinding> keysToRemove = new ArrayList<>();
        for (Map.Entry<KeyBinding, ScheduledFuture<?>> entry : keyReleaseTasks.entrySet()) {
            KeyBinding key = entry.getKey();
            if (exceptions != null && exceptions.contains(key)) {
                continue;
            }
            ScheduledFuture<?> task = entry.getValue();
            if (task != null && !task.isDone()) {
                task.cancel(false);
            }
            if (key != null) {
                key.setPressed(false);
            }
            KeyBinding.updatePressedStates();
            keysToRemove.add(key);
        }

        for (KeyBinding k : keysToRemove) {
            keyReleaseTasks.remove(k);
        }
    }

    private void adjustYaw(ClientPlayerEntity player, float amount) {
        if (player == null) return;
        player.setYaw(player.getYaw() + amount);
    }

    private boolean adjustPitch(ClientPlayerEntity player, float amount) {
        if (player == null) return false;
        float newPitch = player.getPitch() + amount;
        newPitch = Math.max(-80, Math.min(80, newPitch));
        player.setPitch(newPitch);
        return true;
    }

    private Map<String, Object> getPlayerState() {
        Map<String, Object> state = new HashMap<>();
        ClientPlayerEntity player = (client != null) ? client.player : null;
        if (player == null || client == null) {
            // Fallback state
            state.put("x", 0.5);
            state.put("y", 0.5);
            state.put("z", 0.5);
            state.put("yaw_sin", 0.5);
            state.put("yaw_cos", 0.5);
            state.put("pitch", 0.5);
            state.put("health", 1.0);
            state.put("light_level", 0.5);
            state.put("held_item", Arrays.asList(0.8, 0.0));
            state.put("surrounding_blocks", new ArrayList<>(Collections.nCopies(12, 0.5)));
            state.put("mobs", Arrays.asList(0.5, 0.5));
            state.put("results", Arrays.asList(0.0));
            return state;
        }

        try {
            if (Double.isNaN(initialX)) {
                initialX = player.getX();
                initialZ = player.getZ();
            }

            double deltaZ = player.getZ() - initialZ;
            initialZ = player.getZ();

            // x, y, pitch, health, light_level remain as before
            // z = normalized difference (already done similarly to x)
            // x remains as before in original code. The user requested to keep x as it is.
            // So we just copy the logic used for x and y from the old code if needed.
            // The instructions: "keep x,y,pitch,health,light_level,surrounding_blocks as they are"
            // We keep the old approach for x,y exactly.
            double deltaX = player.getX() - initialX;
            initialX = player.getX();

            double encodedX = 0.5;
            if (deltaX > 0) {
                encodedX += Math.min(deltaX / 2.0, 0.5);
            } else if (deltaX < 0) {
                encodedX -= Math.min(-deltaX / 2.0, 0.5);
            }
            encodedX = Math.max(0.0, Math.min(1.0, encodedX));

            double baseY = -64.0;
            double playerY = player.getY();
            double yDiff = playerY - baseY;
            double scaleFactor = 0.05;
            double encodedY = 0.5 + (yDiff * scaleFactor);
            encodedY = Math.max(0.0, Math.min(1.0, encodedY));

            // Z as normalized difference (already handled by old logic)
            double encodedZ = 0.5;
            if (deltaZ > 0) {
                encodedZ += Math.min(deltaZ / 2.0, 0.5);
            } else if (deltaZ < 0) {
                encodedZ -= Math.min(-deltaZ / 2.0, 0.5);
            }
            encodedZ = Math.max(0.0, Math.min(1.0, encodedZ));

            // Compute yaw_sin and yaw_cos
            double rawYaw = player.getYaw();
            // Normalize yaw to [0,360)
            double angleDeg = ((rawYaw % 360.0) + 360.0) % 360.0;
            double angleRad = Math.toRadians(angleDeg);
            double yaw_sin = (Math.sin(angleRad) + 1.0) / 2.0;
            double yaw_cos = (Math.cos(angleRad) + 1.0) / 2.0;

            // pitch as before
            float pitch = player.getPitch();
            pitch = Math.max(-80, Math.min(80, pitch));
            double normalizedPitch = 0.1 + ((pitch + 80) / 160.0) * 0.8;
            normalizedPitch = Math.max(0.0, Math.min(1.0, normalizedPitch));

            // health as before
            double health = Math.max(0, Math.min(player.getHealth(), 20));
            double normalizedHealth = health / 20.0;

            // light_level as before
            int lightLevel = 8; 
            if (client.world != null) {
                lightLevel = client.world.getLightLevel(player.getBlockPos());
            }
            double normalizedLight = Math.max(0, Math.min(lightLevel, 15)) / 15.0;

            // held_item is now fixed
            List<Double> heldItemArray = Arrays.asList(0.8, 0.0);

            // surrounding_blocks keep as they are
            List<Double> surrounding = getDirectionalMobProximity(player);
            // ensure these are normalized between 0 and 1. They already are from logic.

            // mobs: now we only return [lookingAtMob, proximityValue]
            // results: [hitMob]
            double hitMobVal = mobWasHit ? 1.0 : 0.0;
            mobWasHit = false;
            List<Double> mobData = getMobStateData(player); // returns [lookingAtMob, proximityValue]
            List<Double> results = Arrays.asList(hitMobVal);

            state.put("x", encodedX);
            state.put("y", encodedY);
            state.put("z", encodedZ);
            state.put("yaw_sin", yaw_sin);
            state.put("yaw_cos", yaw_cos);
            state.put("pitch", normalizedPitch);
            state.put("health", normalizedHealth);
            state.put("light_level", normalizedLight);
            state.put("held_item", heldItemArray);
            state.put("surrounding_blocks", surrounding);
            state.put("mobs", mobData);
            state.put("results", results);

        } catch (Exception e) {
            System.err.println("[Warning] Error in getPlayerState: " + e.getMessage());
            // Fallback state
            state.clear();
            state.put("x", 0.5);
            state.put("y", 0.5);
            state.put("z", 0.5);
            state.put("yaw_sin", 0.5);
            state.put("yaw_cos", 0.5);
            state.put("pitch", 0.5);
            state.put("health", 1.0);
            state.put("light_level", 0.5);
            state.put("held_item", Arrays.asList(0.8, 0.0));
            state.put("surrounding_blocks", new ArrayList<>(Collections.nCopies(12, 0.5)));
            state.put("mobs", Arrays.asList(0.0, 0.1));
            state.put("results", Arrays.asList(0.0));
        }

        return state;
    }

    private List<Double> getMobStateData(ClientPlayerEntity player) {
        // Previously getMobState() returned [lookingAtMob, hitMob, proximity]
        // Now we only return [lookingAtMob, proximity].
        // hitMob is handled separately.
        double lookingAtMob = 0.0;
        double proximityValue = 0.0;

        if (client != null && client.world != null && player != null) {
            double maxDetectionRange = 20.0;
            double fullValueRange = 2.0;
            double halfValueRange = 8.0;
            double nearestMobDistance = Double.MAX_VALUE;

            Box searchBox = player.getBoundingBox().expand(maxDetectionRange);
            List<Entity> nearbyEntities = client.world.getEntitiesByClass(
                Entity.class,
                searchBox,
                entity -> entity instanceof Monster && entity != player
            );

            for (Entity entity : nearbyEntities) {
                double distance = player.squaredDistanceTo(entity);
                if (distance < nearestMobDistance) {
                    nearestMobDistance = distance;
                }
            }

            nearestMobDistance = Math.sqrt(nearestMobDistance);
            if (nearestMobDistance <= maxDetectionRange) {
                if (nearestMobDistance <= fullValueRange) {
                    proximityValue = 1.0;
                } else if (nearestMobDistance <= halfValueRange) {
                    proximityValue = 1.0 - (0.5 * (nearestMobDistance - fullValueRange) / (halfValueRange - fullValueRange));
                } else {
                    double decayFactor = (maxDetectionRange - nearestMobDistance) / (maxDetectionRange - halfValueRange);
                    proximityValue = 0.5 * Math.pow(decayFactor, 2);
                }
            }

            HitResult hitResult = client.crosshairTarget;
            double reach = 3.5;

            if (hitResult != null && hitResult.getType() == HitResult.Type.ENTITY && player != null) {
                Entity entity = ((EntityHitResult)hitResult).getEntity();
                if (entity instanceof Monster && player.squaredDistanceTo(entity) <= reach * reach) {
                    lookingAtMob = 1.0;
                }
            }
        }

        // Return [lookingAtMob, proximityValue]
        return Arrays.asList(lookingAtMob, proximityValue);
    }

    private List<Double> getDirectionalMobProximity(ClientPlayerEntity player) {
        // This code previously computed directional mob proximity. Keep as is.
        List<Double> directions = new ArrayList<>(Collections.nCopies(12, 0.0));
        if (client == null || client.world == null || player == null) {
            return directions;
        }

        double maxDetectionRange = 20.0;
        double closeRange = 2.0;
        Box searchBox = player.getBoundingBox().expand(maxDetectionRange);

        List<Entity> nearbyEntities = client.world.getEntitiesByClass(
            Entity.class,
            searchBox,
            entity -> entity instanceof Monster && entity != player
        );

        float playerYaw = player.getYaw();
        while (playerYaw > 180) playerYaw -= 360;
        while (playerYaw <= -180) playerYaw += 360;

        for (Entity entity : nearbyEntities) {
            double dx = entity.getX() - player.getX();
            double dz = entity.getZ() - player.getZ();

            double angle = Math.toDegrees(Math.atan2(dz, dx));
            double relativeAngle = (angle - playerYaw + 90);
            while (relativeAngle > 180) relativeAngle -= 360;
            while (relativeAngle <= -180) relativeAngle += 360;

            relativeAngle = (relativeAngle + 360) % 360;
            int directionIndex = ((int)((relativeAngle + 15) / 30)) % 12;

            double distance = Math.sqrt(dx * dx + dz * dz);
            double value = distance <= closeRange ? 1.0 : closeRange / distance;

            if (value > directions.get(directionIndex)) {
                directions.set(directionIndex, Math.min(1.0, value));
            }
        }

        return directions;
    }

    private void sendPlayerState(WebSocket conn) {
        if (!isConnected || conn == null) return;
        try {
            Map<String, Object> state = getPlayerState();
            String stateJson = gson.toJson(state);
            conn.send(stateJson);
        } catch (Exception e) {
            System.err.println("[Warning] Failed to send state: " + e.getMessage());
        }
    }

    private synchronized void scheduleStateSend(WebSocket conn) {
        if (stateSendTask != null && !stateSendTask.isDone()) {
            stateSendTask.cancel(false);
        }

        stateSendTask = scheduler.schedule(() -> {
            sendPlayerState(conn);
        }, STATE_SEND_DELAY, TimeUnit.MILLISECONDS);
    }

    private void executeKeyAction(KeyBinding key, long durationMs, WebSocket conn) {
        if (key == null) {
            scheduleStateSend(conn);
            return;
        }
        ScheduledFuture<?> existingTask = keyReleaseTasks.get(key);
        if (existingTask != null && !existingTask.isDone()) {
            existingTask.cancel(false);
        }

        key.setPressed(true);

        ScheduledFuture<?> releaseTask = scheduler.schedule(() -> {
            key.setPressed(false);
            KeyBinding.updatePressedStates();
            keyReleaseTasks.remove(key);
        }, durationMs, TimeUnit.MILLISECONDS);

        keyReleaseTasks.put(key, releaseTask);
        scheduleStateSend(conn);
    }

    private void executeAttackAction(KeyBinding key, WebSocket conn) {
        if (client == null || client.player == null) {
            scheduleStateSend(conn);
            return;
        }

        HitResult hitResult = null;
        try {
            hitResult = client.player.raycast(20.0, 0.0f, false);
        } catch (Exception e) {
            System.err.println("[Warning] Raycast failed in executeAttackAction: " + e.getMessage());
        }

        if (hitResult != null && hitResult.getType() == HitResult.Type.ENTITY) {
            EntityHitResult entityHit = (EntityHitResult) hitResult;
            if (entityHit.getEntity() instanceof LivingEntity) {
                lastHitMob = entityHit.getEntity();
                mobWasHit = true;
            }
        }

        if (key != null) {
            key.setPressed(true);
            KeyBinding.updatePressedStates();
        }
        scheduleStateSend(conn);
    }

    private void executeJumpWalkForward(KeyBinding jumpKey, KeyBinding forwardKey, WebSocket conn) {
        if (jumpKey == null || forwardKey == null) {
            scheduleStateSend(conn);
            return;
        }

        cancelOngoingActionsExcept(Arrays.asList(jumpKey, forwardKey, (client != null && client.options != null) ? client.options.sneakKey : null, (client != null && client.options != null) ? client.options.attackKey : null));

        jumpKey.setPressed(true);
        scheduler.schedule(() -> {
            jumpKey.setPressed(false);
            KeyBinding.updatePressedStates();
            executeKeyAction(forwardKey, EXTENDED_ACTION_TIME, conn);
        }, 10, TimeUnit.MILLISECONDS);
    }

    private void executeNoOpAction(WebSocket conn) {
        scheduleStateSend(conn);
    }

    private void applyNightVision(Entity player) {
        if (!(player instanceof LivingEntity)) {
            System.err.println("[Warning] Player is not a LivingEntity in applyNightVision.");
            return;
        }
        StatusEffectInstance nightVision = new StatusEffectInstance(StatusEffects.NIGHT_VISION, 20000, 0, false, false);
        ((LivingEntity)player).addStatusEffect(nightVision);
    }

    private void hideUI() {
        if (client != null && client.options != null) {
            client.execute(() -> {
                client.options.hudHidden = true; // Hide the HUD
            });
        }
    }

    private int[] getClientWindowCoordinates() {
        if (client != null && client.getWindow() != null) {
            try {
                return new int[] {
                    client.getWindow().getX(),
                    client.getWindow().getY(),
                    client.getWindow().getWidth(),
                    client.getWindow().getHeight()
                };
            } catch (Exception e) {
                System.err.println("[Warning] Error getting window coordinates: " + e.getMessage());
            }
        }
        return new int[]{0, 0, 800, 600};
    }

    private void cleanupDroppedItems() {
        scheduler.scheduleAtFixedRate(() -> {
            if (client != null && client.world != null) {
                client.execute(() -> {
                    try {
                        List<Entity> toRemove = new ArrayList<>();
                        for (Entity entity : client.world.getEntities()) {
                            if (entity instanceof net.minecraft.entity.ItemEntity) {
                                toRemove.add(entity);
                            }
                        }
                        for (Entity item : toRemove) {
                            if (item != null && item.isAlive()) {
                                item.discard();
                            }
                        }
                    } catch (Exception e) {
                        System.err.println("[Warning] Error cleaning up items: " + e.getMessage());
                    }
                });
            }
        }, 0, 5, TimeUnit.SECONDS);
    }

    private void executeGiveTool(ClientPlayerEntity player, WebSocket conn) {
        if (client != null && client.getServer() != null) {
            client.getServer().execute(() -> {
                if (player != null && player.isAlive()) {
                    ItemStack sword = new ItemStack(Items.DIAMOND_SWORD, 1);
                    player.getInventory().insertStack(sword);
                    System.out.println("gave 1 diamond sword");
                }
            });
        }
        scheduleStateSend(conn);
    }

    private void executeTimedAttackAction(KeyBinding key, WebSocket conn) {
        if (client == null) {
            scheduleStateSend(conn);
            return;
        }

        client.execute(() -> {
            ClientPlayerEntity player = client.player;
            if (player == null || client.interactionManager == null) {
                scheduleStateSend(conn);
                return;
            }

            double reach = 3.5;
            HitResult hitResult = client.crosshairTarget;
            if (hitResult != null && hitResult.getType() == HitResult.Type.ENTITY) {
                EntityHitResult entityHit = (EntityHitResult)hitResult;
                Entity target = entityHit.getEntity();
                if (target instanceof LivingEntity && player.squaredDistanceTo(target) <= reach * reach) {
                    try {
                        client.interactionManager.attackEntity(player, target);
                        player.swingHand(net.minecraft.util.Hand.MAIN_HAND);
                        lastHitMob = target;
                        mobWasHit = true;
                    } catch (Exception e) {
                        System.err.println("[Warning] Failed to attack entity: " + e.getMessage());
                    }
                }
            } else if (hitResult != null && hitResult.getType() == HitResult.Type.BLOCK && key != null) {
                key.setPressed(true);
                KeyBinding.updatePressedStates();
            }
        });

        scheduleStateSend(conn);
    }
}
